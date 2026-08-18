from __future__ import annotations
from dataclasses import dataclass
import cv2, numpy as np
@dataclass(frozen=True)
class ReferenceMotion: common_motion_px:tuple[float,float]; inlier_count:int; residual_px:float; valid:bool; status:str
class ReferenceMonitor:
 def __init__(self,max_corners:int=20): self.max_corners=max_corners
 def estimate(self,previous:np.ndarray,current:np.ndarray,roi:tuple[int,int,int,int])->ReferenceMotion:
  if previous.ndim!=2 or current.shape!=previous.shape: raise ValueError("Reference monitor requires equal grayscale frames")
  x,y,w,h=roi
  if w<3 or h<3 or x<0 or y<0 or x+w>previous.shape[1] or y+h>previous.shape[0]: raise ValueError("Invalid reference ROI")
  mask=np.zeros_like(previous); mask[y:y+h,x:x+w]=255; points=cv2.goodFeaturesToTrack(previous,self.max_corners,.01,5,mask=mask)
  if points is None or len(points)<4:return ReferenceMotion((0,0),0,float("inf"),False,"insufficient_features")
  nextp,st,_=cv2.calcOpticalFlowPyrLK(previous,current,points,None)
  good=st.reshape(-1).astype(bool); src=points[good].reshape(-1,2); dst=nextp[good].reshape(-1,2)
  if len(src)<4:return ReferenceMotion((0,0),len(src),float("inf"),False,"tracking_failed")
  _,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=1.0)
  maski=inliers.reshape(-1).astype(bool) if inliers is not None else np.ones(len(src),bool); flow=dst[maski]-src[maski]; median=np.median(flow,0); return ReferenceMotion(tuple(map(float,median)),int(maski.sum()),float(np.median(np.linalg.norm(flow-median,axis=1))),True,"valid")
