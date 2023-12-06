from .embedder_utils import EmbedderUtils
from typing import Union
from ..utils.misc_enums import MediapipeLandmark, Axes
import numpy as np


class EmbedderVectorByKeyOrNameOperation():
  def __init__(self, point1:Union[int, MediapipeLandmark], point2: Union[int, MediapipeLandmark]):
    self.point1 = EmbedderUtils.resolve_landmark_key(point1)
    self.point2 = EmbedderUtils.resolve_landmark_key(point2)

  def __call__(self, landmarks):
    #print(f"{self.__class__} __call__", self.point1, self.point2)
    return EmbedderUtils.get_vector_by_ids(landmarks, self.point1.value, self.point2.value)

  def get_info(self):
    return f"vector between {self.point1.name} and {self.point2.name}"

class EmbedderVectorHipsCenterShouldersCenterOperation():
  def __call__(self, landmarks):


    point1 = EmbedderUtils.get_average_by_ids(landmarks, MediapipeLandmark.LEFT_HIP.value, MediapipeLandmark.RIGHT_HIP.value)
    #print("point1", point1.shape)
    point2 = EmbedderUtils.get_average_by_ids(landmarks, MediapipeLandmark.LEFT_SHOULDER.value, MediapipeLandmark.RIGHT_SHOULDER.value)
    #print("point2", point2.shape)

    return EmbedderUtils.get_vector(point1,point2)
  def get_info(self):
    return f"vector between HipsCenter and ShouldersCenter"

class EmbedderDistanceByKeyOperation():
  def __init__(self, point1:Union[int, MediapipeLandmark], point2: Union[int, MediapipeLandmark]):
    self.point1 = EmbedderUtils.resolve_landmark_key(point1)
    self.point2 = EmbedderUtils.resolve_landmark_key(point2)
  def __call__(self, landmarks):
    return EmbedderUtils.get_distance_by_ids(landmarks, self.point1.value, self.point2.value)

  def get_info(self):
    return f"Distance  between {self.point1.name} and {self.point2.name}"

class EmbedderAngle3pByKeyOperation():
  def __init__(self,  point1:Union[int, MediapipeLandmark], point2: Union[int, MediapipeLandmark], point3: Union[int, MediapipeLandmark]):
    
    self.point1 = EmbedderUtils.resolve_landmark_key(point1)
    self.point2 = EmbedderUtils.resolve_landmark_key(point2)
    self.point3 = EmbedderUtils.resolve_landmark_key(point3)
    
  def __call__(self, landmarks):
    return EmbedderUtils.get_cosine_angle_by_ids_3p(landmarks, self.point1.value, self.point2.value, self.point3.value)
 
  def get_info(self):
    return f"The angle  at {self.point2.name} between  {self.point1.name} and {self.point3.name}"

class EmbedderAngle4pByKeyOperation():
  def __init__(self,  point1_1:Union[int, MediapipeLandmark], point1_2: Union[int, MediapipeLandmark], point2_1: Union[int, MediapipeLandmark], point2_2: Union[int, MediapipeLandmark]):
    self.point1_1 = EmbedderUtils.resolve_landmark_key(point1_1)
    self.point1_2 = EmbedderUtils.resolve_landmark_key(point1_2)
    self.point2_1 = EmbedderUtils.resolve_landmark_key(point2_1)
    self.point2_2 = EmbedderUtils.resolve_landmark_key(point2_2)
    
  def __call__(self, landmarks):
    return EmbedderUtils.get_cosine_angle_between_vectors(
              EmbedderUtils.get_vector_by_ids(landmarks,self.point1_1.value, self.point1_2.value),
              EmbedderUtils.get_vector_by_ids(landmarks,self.point2_1.value, self.point2_2.value )
            ) 
  def get_info(self):
    return f"The angle  between segment {self.point1_1.name}-{self.point1_2.name} and segment {self.point2_1.name}-{self.point2_2.name}"


class EmbedderAngleVectorByKeyAndAxe():
  def __init__(self,  point1:Union[int, MediapipeLandmark], point2: Union[int, MediapipeLandmark], axe:Axes = Axes.X ):
    self.point1 = EmbedderUtils.resolve_landmark_key(point1)
    self.point2 = EmbedderUtils.resolve_landmark_key(point2)
    self.axe = axe

  def __call__(self, landmarks):

     return EmbedderUtils.get_cosine_angle_between_vectors(
              EmbedderUtils.get_vector_by_ids(landmarks, self.point1.value, self.point2.value),
              EmbedderUtils.get_vector(np.zeros(3),np.array(self.axe.value))
      )
      

  def get_info(self):
    return f"The angle  between segment {self.point1.name}-{self.point2.name} and axe {self.axe.name} "