"""Group-OOF sigmoid calibration for binary risk and 3-class hazard subtype."""
import numpy as np
from sklearn.linear_model import LogisticRegression

class BinarySigmoidCalibrator:
    calibrated=True
    def __init__(self): self.model=None
    def fit(self, scores, y):
        x=np.asarray(scores,dtype=float).reshape(-1,1); y=np.asarray(y,dtype=int)
        if len(x)<20 or len(np.unique(y))<2: raise ValueError("Need >=20 OOF samples and both classes for calibration")
        self.model=LogisticRegression(C=1.0,solver="lbfgs",max_iter=2000).fit(x,y)
        return self
    def predict(self, scores):
        if self.model is None: raise RuntimeError("Calibrator is not fitted")
        return self.model.predict_proba(np.asarray(scores,dtype=float).reshape(-1,1))[:,1]

class MulticlassSigmoidCalibrator:
    calibrated=True
    """One-vs-rest Platt scaling followed by probability normalization."""
    def __init__(self, classes):
        self.classes_=np.asarray(list(classes),dtype=object)
        self.models=[]
    def fit(self, probabilities, y):
        p=np.asarray(probabilities,dtype=float); y=np.asarray(y,dtype=object)
        if p.ndim!=2 or p.shape[1]!=len(self.classes_): raise ValueError("Probability shape/classes mismatch")
        if len(p)<30: raise ValueError("Need at least 30 grouped OOF hazard samples")
        self.models=[]
        for cls in self.classes_:
            target=(y==cls).astype(int)
            if len(np.unique(target))<2: raise ValueError(f"Calibration OOF data lacks positive/negative examples for {cls}")
            model=LogisticRegression(C=1.0,solver="lbfgs",max_iter=2000)
            model.fit(p[:,[len(self.models)]],target)
            self.models.append(model)
        return self
    def predict(self, probabilities):
        if not self.models: raise RuntimeError("Calibrator is not fitted")
        p=np.asarray(probabilities,dtype=float)
        out=np.column_stack([m.predict_proba(p[:,[i]])[:,1] for i,m in enumerate(self.models)])
        sums=out.sum(axis=1,keepdims=True)
        return np.divide(out,sums,out=np.full_like(out,1.0/out.shape[1]),where=sums>0)
