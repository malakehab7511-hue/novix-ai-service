"""Final NOVIX hybrid training, grouped-OOF calibration, and LOEO evaluation."""
import json, argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, balanced_accuracy_score,
                             brier_score_loss, precision_recall_fscore_support)
from .config import RAW_DATA, PROCESSED_DATA, MODELS_DIR, REPORTS_DIR, FEATURES, TARGET, GROUP_ENV, GROUP_SESSION, SAFE_LABELS, ALL_LABELS, N_CONSECUTIVE, MAX_GAP_SECONDS
from .data_processing import load_and_clean
from .models import make_binary_model, make_multiclass_model
from .evaluation import _positive_probability, choose_balanced_threshold, consecutive_alarm, tune_threshold_oof
from .probability_calibration import BinarySigmoidCalibrator, MulticlassSigmoidCalibrator

UNKNOWN="HAZARD_UNKNOWN"
HAZARD_CLASSES=["GAS_LEAK","SMOKE_LIKE","FIRE"]
MIN_SUBTYPE_SCORE=0.50
MIN_SUBTYPE_MARGIN=0.10


def session_groups(df):
    return (df[GROUP_ENV].astype(str)+"::"+df[GROUP_SESSION].astype(str)).to_numpy()


def grouped_binary_oof(X,y,groups,model_name):
    X=X.reset_index(drop=True); y=pd.Series(y).reset_index(drop=True); groups=np.asarray(groups)
    oof=np.full(len(y),np.nan)
    n=min(5,len(np.unique(groups)))
    for tr,va in GroupKFold(n_splits=n).split(X,y,groups):
        if len(np.unique(y.iloc[tr]))<2: continue
        m=make_binary_model(model_name); m.fit(X.iloc[tr],y.iloc[tr])
        oof[va]=_positive_probability(m,X.iloc[va])
    valid=np.isfinite(oof)
    return oof,valid


def grouped_hazard_oof(X,y,groups,model_name):
    X=X.reset_index(drop=True); y=pd.Series(y).reset_index(drop=True); groups=np.asarray(groups)
    oof=np.full((len(y),len(HAZARD_CLASSES)),np.nan)
    n=min(5,len(np.unique(groups)))
    for tr,va in GroupKFold(n_splits=n).split(X,y,groups):
        if set(y.iloc[tr].unique()) != set(HAZARD_CLASSES): continue
        m=make_multiclass_model(model_name); m.fit(X.iloc[tr],y.iloc[tr])
        p=m.predict_proba(X.iloc[va]); classes=list(m.classes_)
        oof[va]=p[:,[classes.index(c) for c in HAZARD_CLASSES]]
    valid=np.isfinite(oof).all(axis=1)
    if valid.sum()<30 or set(y[valid].unique()) != set(HAZARD_CLASSES):
        raise RuntimeError(f"Insufficient grouped OOF hazard coverage: {valid.sum()} rows, classes={sorted(y[valid].unique())}")
    return oof,valid


def ece_toplabel(probabilities,y_true,classes,bins=10):
    p=np.asarray(probabilities,float); y=np.asarray(y_true,dtype=object)
    conf=p.max(axis=1); pred=np.asarray(classes,dtype=object)[p.argmax(axis=1)]
    edges=np.linspace(0,1,bins+1); ece=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if mask.any(): ece += mask.mean()*abs(np.mean(pred[mask]==y[mask])-np.mean(conf[mask]))
    return float(ece)


def multiclass_brier(p,y,classes):
    p=np.asarray(p,float); y=np.asarray(y,dtype=object)
    one=np.column_stack([(y==c).astype(float) for c in classes])
    return float(np.mean(np.sum((p-one)**2,axis=1)))


class IdentityBinaryCalibrator:
    calibrated=False
    def predict(self,scores): return np.asarray(scores,dtype=float)
class IdentityMulticlassCalibrator:
    calibrated=False
    def __init__(self,classes): self.classes_=np.asarray(classes,dtype=object)
    def predict(self,probabilities): return np.asarray(probabilities,dtype=float)

def fit_calibrators(train,model_name,strict=False):
    X=train[FEATURES].astype(float).reset_index(drop=True)
    y=train[TARGET].astype(str).reset_index(drop=True)
    groups=session_groups(train)
    ybin=(~y.isin(SAFE_LABELS)).astype(int)
    try:
        bin_oof,bin_valid=grouped_binary_oof(X,ybin,groups,model_name)
        if bin_valid.sum()<30 or ybin.to_numpy()[bin_valid].min()==ybin.to_numpy()[bin_valid].max():
            raise ValueError("Grouped OOF binary coverage does not contain both classes")
        bin_cal=BinarySigmoidCalibrator().fit(bin_oof[bin_valid],ybin.to_numpy()[bin_valid])
        bin_rows=int(bin_valid.sum())
    except Exception:
        if strict: raise
        bin_cal=IdentityBinaryCalibrator(); bin_rows=0
    hazard_mask=~y.isin(SAFE_LABELS)
    Xh=X.loc[hazard_mask].reset_index(drop=True); yh=y.loc[hazard_mask].reset_index(drop=True)
    gh=groups[hazard_mask.to_numpy()]
    try:
        haz_oof,haz_valid=grouped_hazard_oof(Xh,yh,gh,model_name)
        haz_cal=MulticlassSigmoidCalibrator(HAZARD_CLASSES).fit(haz_oof[haz_valid],yh.to_numpy()[haz_valid])
        hrows=int(haz_valid.sum())
        raw_brier=multiclass_brier(haz_oof[haz_valid],yh.to_numpy()[haz_valid],HAZARD_CLASSES)
        cal_brier=multiclass_brier(haz_cal.predict(haz_oof[haz_valid]),yh.to_numpy()[haz_valid],HAZARD_CLASSES)
        cal_ece=ece_toplabel(haz_cal.predict(haz_oof[haz_valid]),yh.to_numpy()[haz_valid],HAZARD_CLASSES)
    except Exception:
        if strict: raise
        haz_cal=IdentityMulticlassCalibrator(HAZARD_CLASSES); hrows=0
        raw_brier=cal_brier=cal_ece=np.nan
    return bin_cal,haz_cal,{
      "binary_oof_rows":bin_rows,"hazard_subtype_oof_rows":hrows,
      "hazard_oof_brier_raw":raw_brier,
      "hazard_oof_brier_calibrated_fit_diagnostic":cal_brier,
      "hazard_oof_ece_calibrated_fit_diagnostic":cal_ece,
      "binary_calibrated":bool(getattr(bin_cal,"calibrated",True)),
      "subtype_calibrated":bool(getattr(haz_cal,"calibrated",True)),
      "calibration_note":"OOF fit diagnostics are not an independent test; outer held-out environment metrics are the generalization check. If grouped OOF class coverage is insufficient, identity passthrough is used and calibration_status is false."}


def predict_fold(train,test,model_name,save_reports=True,fold=""):
    Xtr=train[FEATURES].astype(float); Xte=test[FEATURES].astype(float)
    ytr=train[TARGET].astype(str); yte=test[TARGET].astype(str)
    ybin=(~ytr.isin(SAFE_LABELS)).astype(int)
    bin_cal,haz_cal,cal_info=fit_calibrators(train,model_name)
    # Tune the 10-reading decision threshold using grouped-OOF calibrated gate scores.
    groups_tr=session_groups(train)
    raw_gate_oof,gate_valid=grouped_binary_oof(Xtr,ybin,groups_tr,model_name)
    if gate_valid.sum()>=30 and ybin.to_numpy()[gate_valid].min()!=ybin.to_numpy()[gate_valid].max():
        calibrated_gate_oof=bin_cal.predict(raw_gate_oof[gate_valid])
        th_info=choose_balanced_threshold(ybin.to_numpy()[gate_valid],calibrated_gate_oof,
                     groups_tr[gate_valid],train.elapsed_s.to_numpy()[gate_valid])
        threshold=float(th_info["threshold"])
    else:
        threshold,th_info=tune_threshold_oof(Xtr.reset_index(drop=True),ybin.reset_index(drop=True),groups_tr,
                                             train.elapsed_s.to_numpy(),model_name)
    # Fit final heads on the complete outer training fold.
    gate=make_binary_model(model_name).fit(Xtr,ybin)
    hm=~ytr.isin(SAFE_LABELS); sm=ytr.isin(SAFE_LABELS)
    hazard=make_multiclass_model(model_name).fit(Xtr.loc[hm],ytr.loc[hm])
    safe=(make_multiclass_model(model_name) if ytr.loc[sm].nunique()>1 else DummyClassifier(strategy="most_frequent"))
    safe.fit(Xtr.loc[sm],ytr.loc[sm])

    raw_p= _positive_probability(gate,Xte)
    p_h=bin_cal.predict(raw_p)
    gte=session_groups(test)
    alarm_gate=consecutive_alarm(p_h,gte,test.elapsed_s.to_numpy(),threshold)
    flame=test.f_flame.to_numpy(float)>=1.0
    pred=np.asarray(safe.predict(Xte),dtype=object).reshape(-1)
    raw_sub=hazard.predict_proba(Xte)
    hclasses=list(hazard.classes_)
    raw_sub=raw_sub[:,[hclasses.index(c) for c in HAZARD_CLASSES]]
    cal_sub=haz_cal.predict(raw_sub)
    top=cal_sub.argmax(axis=1); score=cal_sub.max(axis=1)
    sortedp=np.sort(cal_sub,axis=1); margin=sortedp[:,-1]-sortedp[:,-2]
    subtype=np.asarray(HAZARD_CLASSES,dtype=object)[top]
    subtype[(score<MIN_SUBTYPE_SCORE)|(margin<MIN_SUBTYPE_MARGIN)]=UNKNOWN
    pred[alarm_gate]=subtype[alarm_gate]
    pred[flame]="FIRE"
    detected=alarm_gate|flame
    ytrue=yte.to_numpy(dtype=object)
    true_haz=~np.isin(ytrue,list(SAFE_LABELS)); safe_rows=~true_haz
    pred_haz=np.isin(pred,HAZARD_CLASSES+[UNKNOWN])
    labels=ALL_LABELS+[UNKNOWN]
    rep=classification_report(ytrue,pred,labels=labels,output_dict=True,zero_division=0)
    cm=confusion_matrix(ytrue,pred,labels=labels)
    REPORTS_DIR.mkdir(parents=True,exist_ok=True)
    if save_reports:
        pd.DataFrame(cm,index=labels,columns=labels).to_csv(REPORTS_DIR/f"final_confusion_{model_name}_{fold}.csv")
        pd.DataFrame(rep).T.to_csv(REPORTS_DIR/f"final_classification_{model_name}_{fold}.csv")
    # Independent held-out hazard subtype calibration scores.
    hazard_rows=true_haz
    raw_brier=cal_brier=cal_ece=np.nan
    if hazard_rows.any():
        yt_h=ytrue[hazard_rows]
        raw_brier=multiclass_brier(raw_sub[hazard_rows],yt_h,HAZARD_CLASSES)
        cal_brier=multiclass_brier(cal_sub[hazard_rows],yt_h,HAZARD_CLASSES)
        cal_ece=ece_toplabel(cal_sub[hazard_rows],yt_h,HAZARD_CLASSES)
    metrics={
      "model":model_name,"held_out_environment":fold,"test_rows":len(test),"threshold_calibrated_gate":threshold,
      "hazard_recall_with_flame_override":float(detected[true_haz].mean()) if true_haz.any() else np.nan,
      "hazard_recall_ai_gate_only":float(alarm_gate[true_haz].mean()) if true_haz.any() else np.nan,
      "safe_false_alarm_fraction":float(pred_haz[safe_rows].mean()) if safe_rows.any() else np.nan,
      "five_class_macro_f1_including_unknown":f1_score(ytrue,pred,labels=ALL_LABELS,average="macro",zero_division=0),
      "five_class_weighted_f1":f1_score(ytrue,pred,labels=ALL_LABELS,average="weighted",zero_division=0),
      "subtype_accuracy_on_detected_hazards":float(np.mean(pred[detected&true_haz]==ytrue[detected&true_haz])) if np.any(detected&true_haz) else np.nan,
      "unknown_fraction_on_detected_hazards":float(np.mean(pred[detected&true_haz]==UNKNOWN)) if np.any(detected&true_haz) else np.nan,
      "flame_override_rows":int(flame.sum()),"hazard_subtype_brier_raw":raw_brier,
      "hazard_subtype_brier_calibrated":cal_brier,"hazard_subtype_ece_calibrated":cal_ece,
      "recall_NORMAL":rep.get("NORMAL",{}).get("recall",np.nan),
      "recall_BENIGN_INTERFERENCE":rep.get("BENIGN_INTERFERENCE",{}).get("recall",np.nan),
      "recall_GAS_LEAK":rep.get("GAS_LEAK",{}).get("recall",np.nan),
      "recall_SMOKE_LIKE":rep.get("SMOKE_LIKE",{}).get("recall",np.nan),
      "recall_FIRE":rep.get("FIRE",{}).get("recall",np.nan),
      **{f"calibration_{k}":v for k,v in cal_info.items() if isinstance(v,(int,float))}}
    return metrics


def train_final_bundle(df,model_name):
    bin_cal,haz_cal,cal_info=fit_calibrators(df,model_name,strict=True)
    X=df[FEATURES].astype(float); y=df[TARGET].astype(str); ybin=(~y.isin(SAFE_LABELS)).astype(int)
    groups=session_groups(df)
    oof,valid=grouped_binary_oof(X,ybin,groups,model_name)
    pcal=bin_cal.predict(oof[valid])
    threshold_info=choose_balanced_threshold(ybin.to_numpy()[valid],pcal,groups[valid],df.elapsed_s.to_numpy()[valid])
    gate=make_binary_model(model_name).fit(X,ybin)
    hm=~y.isin(SAFE_LABELS); sm=y.isin(SAFE_LABELS)
    hazard=make_multiclass_model(model_name).fit(X.loc[hm],y.loc[hm])
    safe=make_multiclass_model(model_name).fit(X.loc[sm],y.loc[sm])
    bundle={"version":"3.0-final-raw-stream-hybrid","model_name":model_name,
      "stage1_hazard_gate":gate,"stage2_safe_classifier":safe,"stage2_hazard_classifier":hazard,
      "hazard_gate_calibrator":bin_cal,"hazard_subtype_calibrator":haz_cal,
      "threshold":float(threshold_info["threshold"]),"threshold_info":threshold_info,
      "features":FEATURES,"safe_labels":sorted(SAFE_LABELS),"hazard_labels":HAZARD_CLASSES,
      "unknown_label":UNKNOWN,"min_subtype_score":MIN_SUBTYPE_SCORE,"min_subtype_margin":MIN_SUBTYPE_MARGIN,
      "n_consecutive":N_CONSECUTIVE,"max_gap_seconds":MAX_GAP_SECONDS,
      "feature_equations_version":"NOVIX_RAW_FEATURE_EQUATIONS_2026-11-06",
      "calibration_info":cal_info,
      "safety_note":"Software flame override is immediate. Physical flame/gas safety interlocks must remain independent of this AI process."}
    MODELS_DIR.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,MODELS_DIR/f"{model_name.lower()}_hybrid_final_v3.joblib")
    return bundle


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--models",nargs="+",choices=["LightGBM"],default=["LightGBM"])
    args=parser.parse_args()
    df,audit=load_and_clean(RAW_DATA,PROCESSED_DATA)
    (REPORTS_DIR/"final_data_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    rows=[]
    for name in args.models:
        print(f"\n=== FINAL {name}: Leave-One-Environment-Out ===")
        for env in sorted(df[GROUP_ENV].unique()):
            tr=df[df[GROUP_ENV]!=env].reset_index(drop=True)
            te=df[df[GROUP_ENV]==env].reset_index(drop=True)
            m=predict_fold(tr,te,name,True,str(env))
            rows.append(m)
            print({k:m[k] for k in ["held_out_environment","hazard_recall_with_flame_override","safe_false_alarm_fraction","five_class_macro_f1_including_unknown","recall_GAS_LEAK","recall_SMOKE_LIKE","recall_FIRE","hazard_subtype_brier_raw","hazard_subtype_brier_calibrated"]})
        train_final_bundle(df,name)
    out=pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR/"final_loeo_evaluation.csv",index=False)
    # Compact mean only across environments where a metric is defined.
    numeric=out.select_dtypes(include=[np.number]).mean(numeric_only=True).to_frame("mean_across_environment_folds")
    numeric.to_csv(REPORTS_DIR/"final_loeo_mean_metrics.csv")
    print("\nSaved final model bundle and LOEO evaluation reports.")

if __name__=="__main__": main()
