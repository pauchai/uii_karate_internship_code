from typing import Union
from ..utils.misc_enums import MediapipeLandmark
import numpy as np 

class EmbedderUtils():
    @classmethod
    def resolve_landmark_key(cls, key_of_landmark: Union[MediapipeLandmark, int])->int :
      #print(f"{cls} resolve_landmark_key", key_of_landmark)
      if isinstance(key_of_landmark, MediapipeLandmark):
          landmark_value = key_of_landmark
      elif isinstance(key_of_landmark, int):
          # Use the raw value for int
          landmark_value = MediapipeLandmark.get_member_by_id(key_of_landmark)

      else:
          # Handle other cases or raise an error
          raise ValueError("Invalid type for key_from")
      #print(f"{cls} resolve_landmark_key return ", key_of_landmark)

      return landmark_value
    @classmethod
    def get_joint_center_by_ids(cls, landmarks, key_from, key_to):
      """Calculates center between keys."""
      return cls.get_average_by_ids(landmarks, key_from, key_to)

    @classmethod
    def get_pose_center(cls, landmarks):
      """Calculates pose center as point between hips."""
      return cls.get_joint_center_by_ids(landmarks, MediapipeLandmark.LEFT_HIP.value, MediapipeLandmark.RIGHT_HIP.value)

    @classmethod
    def get_pose_size(cls, landmarks, torso_size_multiplier):
      """Calculates pose size.

      It is the maximum of two values:
        * Torso size multiplied by `torso_size_multiplier`
        * Maximum distance from pose center to any pose landmark
      """
      # This approach uses only 2D landmarks to compute pose size.
      landmarks = landmarks[:, :2]

      # Hips center.
      hips =  cls.get_joint_center_by_ids(landmarks, MediapipeLandmark.LEFT_HIP.value, MediapipeLandmark.RIGHT_HIP.value)

      # Shoulders center.
      shoulders =  cls.get_joint_center_by_ids(landmarks, MediapipeLandmark.LEFT_SHOULDER.value, MediapipeLandmark.RIGHT_SHOULDER.value)

      # Torso size as the minimum body size.
      torso_size = np.linalg.norm(shoulders - hips)

      # Max dist to pose center.
      pose_center = cls.get_pose_center(landmarks)
      max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))
      return max(torso_size * torso_size_multiplier, max_dist)

    @classmethod
    def get_average_by_ids(cls, landmarks, id_from: int, id_to: int):
      #print("get_average_by_ids", id_from, id_to)
      lmk_from = landmarks[id_from]
      lmk_to = landmarks[id_to]
      return (lmk_from + lmk_to) * 0.5

    @classmethod
    def get_vector_by_ids(cls, landmarks, id_from:int, id_to:int):
      #print("id_from", id_from)
      lmk_from = landmarks[id_from]
      #print(lmk_from.shape)
      lmk_to = landmarks[id_to]
      return cls.get_vector(lmk_from, lmk_to)

    @classmethod
    def get_cosine_angle_by_ids_3p(cls, landmarks, id_from:int, id_center:int, id_to:int):
      lmk_from = landmarks[id_from]
      lmk_center = landmarks[id_center]
      lmk_to = landmarks[id_to]
      return cls.get_cosine_angle_3p(lmk_from, lmk_center, lmk_to)

    @classmethod
    def get_cosine_angle_3p(cls, lmk_from, lmk_center, lmk_to):
      vector1 = cls.get_vector(lmk_center, lmk_from)
      vector2 = cls.get_vector(lmk_center, lmk_to)
      return cls.get_cosine_angle_between_vectors(vector1, vector2)
    
    @classmethod
    def get_cosine_angle_between_vectors(cls, vector1, vector2):
      # Calculate the dot product and magnitudes
      dot_product = np.dot(vector1, vector2)
      magnitude_vector1 = np.linalg.norm(vector1)
      magnitude_vector2 = np.linalg.norm(vector2)

      # Calculate the cosine of the angle
      cosine_angle = dot_product / (magnitude_vector1 * magnitude_vector2)

      return cosine_angle

    @classmethod
    def get_distance_by_ids(cls, landmarks, id_from:int, id_to:int):
      lmk_from = landmarks[id_from]
      lmk_to = landmarks[id_to]
      return cls.get_distance(lmk_from, lmk_to)

    @classmethod
    def get_distance(cls, lmk_from, lmk_to):
      return np.linalg.norm(cls.get_vector(lmk_from , lmk_to))


    @classmethod
    def get_vector(cls, lmk_from, lmk_to):
      return lmk_to - lmk_from
