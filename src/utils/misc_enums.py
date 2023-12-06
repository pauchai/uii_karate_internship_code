import enum
class Axes(enum.Enum):
  X = [1,0,0]
  Y = [0,1,0]
  Z = [0,0,1]
  
class MediapipeLandmark(enum.Enum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY_1 = 17
    RIGHT_PINKY_1 = 18
    LEFT_INDEX_1 = 19
    RIGHT_INDEX_1 = 20
    LEFT_THUMB_2 = 21
    RIGHT_THUMB_2 = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    @classmethod
    def get_member_by_id(cls, id):
      #print(f"{cls.__class__}", id)
      for name, member in cls.__members__.items():
          if member.value == id:
                #print(f"{cls.__class__}", member)
                return member
      return None
    @classmethod
    def get_name_by_id(cls, id):
        if isinstance(id, cls):
          return id.name

        for name, member in cls.__members__.items():
            if member.value == id or member.name == id :
                return name
        return None