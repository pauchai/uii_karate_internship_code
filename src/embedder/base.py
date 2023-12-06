class FullBodyPoseEmbedder(object):
  """Converts 3D pose landmarks into 3D embedding."""

  def __init__(self, torso_size_multiplier=2.5):
    # Multiplier to apply to the torso to get minimal body size.
    self._torso_size_multiplier = torso_size_multiplier
    self._embeddings_operations = None
    self.build()

  def get_info(self):
    embedding_info = []
    for operation in self._embeddings_operations:
      embedding_info.append(operation.get_info())
    return embedding_info

  def __call__(self, landmarks):
    """Normalizes pose landmarks and converts to embedding

    Args:
      landmarks - NumPy array with 3D landmarks of shape (N, 3).

    Result:
      Numpy array with pose embedding of shape (M, 3) where `M` is the number of
      pairwise distances defined in `get_pose_distance_embedding`.
    """
    #assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(landmarks.shape[0])

    # Get pose landmarks.
    landmarks = np.copy(landmarks)

    # Normalize landmarks.
    #landmarks = self._normalize_pose_landmarks(landmarks)

    # Get embedding.
    embedding = self._get_pose_vector_embedding(landmarks)

    return embedding

  def _normalize_pose_landmarks(self, landmarks):
    """Normalizes landmarks translation and scale."""
    landmarks = np.copy(landmarks)

    # Normalize translation.
    pose_center = EmbedderUtils.get_pose_center(landmarks)
    landmarks -= pose_center

    # Normalize scale.
    pose_size = EmbedderUtils.get_pose_size(landmarks, self._torso_size_multiplier)
    landmarks /= pose_size
    # Multiplication by 100 is not required, but makes it eaasier to debug.
    #landmarks *= 100

    return landmarks

  def build(self):
    self._embeddings_operations = [
        EmbedderVectorHipsCenterShouldersCenterOperation(),
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.LEFT_SHOULDER, MediapipeLandmark.LEFT_ELBOW),
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.RIGHT_SHOULDER, MediapipeLandmark.RIGHT_ELBOW),
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.LEFT_ELBOW, MediapipeLandmark.LEFT_WRIST),
        EmbedderVectorByKeyOrNameOperation( MediapipeLandmark.RIGHT_ELBOW, MediapipeLandmark.RIGHT_WRIST),#4
        EmbedderVectorByKeyOrNameOperation( MediapipeLandmark.LEFT_HIP, MediapipeLandmark.LEFT_KNEE),#5
        EmbedderVectorByKeyOrNameOperation( MediapipeLandmark.RIGHT_HIP, MediapipeLandmark.RIGHT_KNEE),#6
        EmbedderVectorByKeyOrNameOperation( MediapipeLandmark.LEFT_KNEE, MediapipeLandmark.LEFT_ANKLE),#7
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.RIGHT_KNEE,MediapipeLandmark.RIGHT_ANKLE),#8
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.LEFT_SHOULDER, MediapipeLandmark.LEFT_WRIST),#9
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.RIGHT_SHOULDER,MediapipeLandmark.RIGHT_WRIST),#10
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.LEFT_HIP, MediapipeLandmark.LEFT_ANKLE),#11
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.RIGHT_HIP, MediapipeLandmark.RIGHT_ANKLE),#12
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.LEFT_HIP, MediapipeLandmark.LEFT_WRIST),#13
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.RIGHT_HIP, MediapipeLandmark.RIGHT_WRIST),#14
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.LEFT_SHOULDER, MediapipeLandmark.LEFT_ANKLE),#15
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.RIGHT_SHOULDER, MediapipeLandmark.RIGHT_ANKLE),#16
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.LEFT_HIP, MediapipeLandmark.LEFT_WRIST),#17
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.RIGHT_HIP,MediapipeLandmark.RIGHT_WRIST),#18
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.LEFT_ELBOW,MediapipeLandmark.RIGHT_ELBOW),#19
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.LEFT_KNEE, MediapipeLandmark.RIGHT_KNEE),#20
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.LEFT_WRIST,MediapipeLandmark.RIGHT_WRIST),#21
        EmbedderVectorByKeyOrNameOperation(MediapipeLandmark.LEFT_ANKLE,MediapipeLandmark.RIGHT_ANKLE),#22
    ]



  def _get_pose_vector_embedding(self, landmarks):
    """Converts pose landmarks into 3D embedding.

    We use several pairwise 3D distances to form pose embedding. All distances
    include X and Y components with sign. We differnt types of pairs to cover
    different pose classes. Feel free to remove some or add new.

    Args:
      landmarks - NumPy array with 3D landmarks of shape (N, 3).

    Result:
      Numpy array with pose embedding of shape (M, 3) where `M` is the number of
      pairwise distances.
    """
    embedding = []
    for embedding_operation in self._embeddings_operations:
      embedding.append(embedding_operation(landmarks))

    return np.array(embedding)

