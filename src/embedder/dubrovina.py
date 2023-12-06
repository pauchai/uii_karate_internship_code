class DubrovinaEmbedder(FullBodyPoseEmbedder):

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
                                           
  def __init__(self, torso_size_multiplier=1):
    self._embeddings_operations = None
    self.distances = [
      (24, 28), (24, 26), (26, 28),  # Расстояния между ключевыми точками правой ноги
      (12, 16), (12, 14), (14, 16),  # Расстояния между ключевыми точками правой руки
      (23, 27), (23, 25), (25, 27),  # Расстояния между ключевыми точки левой ноги
      (11, 15), (11, 13), (13, 15),  # Расстояния между ключевыми точками левой руки
      (13, 14),  # Расстояние между локтями двух рук
      (15, 16),  # Расстояние между запястьями двух рук
      (25, 26),  # Расстояние между коленями двух ног
      (27, 28),  # Расстояние между лодыжками двух ног
      (14, 26),  # Расстояние между правым локтем и правым коленом
      (13, 25),  # Расстояние между левым локтем и левым коленом
      (14, 25),  # Расстояние между правым локтем и левым коленом
      (13, 26),  # Расстояние между левым локтем и правым коленом
      (12, 26),  # Расстояние между правым плечом и правым коленом
      (11, 25),  # Расстояние между левым плечом и левым коленом
      (11, 26),  # Расстояние между левым плечом и правым коленом
      (12, 25),  # Расстояние между правым плечом и левым коленом
    ]

    # Зададим тройки точек, между которыми нужно расчитать угол (B-A-C)
    self.angles = [
        (12, 14, 16),  # угол в правом локтевом суставе
        (11, 13, 15),  # угол в левом локтевом суставе
        (14, 12, 24),  # угол в правом плечевом суставе (вертикально)
        (13, 11, 23),  # угол в левом плечевом суставе (вертикально)
        (14, 12, 11),  # угол в правом плечевом суставе (горизонтально)
        (13, 11, 12),  # угол в левом плечевом суставе (горизонтально)
        (24, 26, 28),  # угол в правом коленном суставе
        (23, 25, 27),  # угол в левом коленном суставе
        (12, 24, 26),  # угол в правом тазобедренном суставе
        (11, 23, 25),  # угол в левом тазобедренном суставе
    ]

    # Зададим две пары точек, задающие скрещивающиеся прямые (M-N) и (K-L), между которыми нужно расчитать угол
    self.gammas = [
        (12, 11, 28, 27),  # угол между линией плеч и линией щиколоток
        (12, 11, 24, 23),  # угол между линией плеч и линией таза
        (12, 11, 26, 25),  # угол между линией плеч и линией колен
        (12, 11, 14, 13),  # угол между линией плеч и линией локтей
        (12, 11, 16, 15),  # угол между линией плеч и линией запястий
        (14, 16, 13, 15),  # угол между линией локоть-запястье правой руки и левой руки
        (12, 14, 11, 13),  # угол между линией плечо-локоть правой руки и левой руки
        (24, 26, 23, 25),  # угол между линией бедро-колено правой ноги и левой ноги
        (26, 28, 25, 27),  # угол между линией колено-щиколотка правой ноги и левой ноги
    ]

    # Зададим пары точек, задающие прямую (M-N), между которой нужно расчитать углы к осям OY, OX, OZ
    self.omegas = [
        (12, 14),  # линия плечо-локоть правой руки
        (11, 13),  # линия плечо-локоть левой руки
        (14, 16),  # линия локоть-запястье правой руки
        (13, 15),  # линия локоть-запястье левой руки
        (24, 26),  # линия бедро-колено правой ноги
        (23, 25),  # линия бедро-колено левой ноги
        (26, 28),  # линия колено-щиколотка правой ноги
        (25, 27),  # линия колено-щиколотка левой ноги
    ]
    self.build()

  def build(self):

    self._embeddings_operations = []

    for point1_id, point2_id in self.distances:
        self._embeddings_operations.append(EmbedderDistanceByKeyOperation(point1_id, point2_id))
    
    for point1_id, point2_id, point3_id in self.angles:
        #print(point1_id, point2_id, point3_id)
        self._embeddings_operations.append(EmbedderAngle3pByKeyOperation( point1_id, point2_id, point3_id))
    
    for point1_1_id, point1_2_id, point2_1_id, point2_2 in self.gammas:
        self._embeddings_operations.append(EmbedderAngle4pByKeyOperation( point1_1_id, point1_2_id, point2_1_id, point2_2))

    
    for point1_id, point2_id in self.omegas:
        self._embeddings_operations.append(EmbedderAngleVectorByKeyAndAxe( point1_id, point2_id, Axes.Y))
        self._embeddings_operations.append(EmbedderAngleVectorByKeyAndAxe( point1_id, point2_id, Axes.X))
        self._embeddings_operations.append(EmbedderAngleVectorByKeyAndAxe( point1_id, point2_id, Axes.Z))
        
    
