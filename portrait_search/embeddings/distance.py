from portrait_search.core.enums import DistanceType

DISTANCE_TO_SIMILARITY = {
    DistanceType.COSINE: lambda x: x,
    DistanceType.EUCLIDEAN: lambda x: 1 / (1 + x),
    DistanceType.DOT_PRODUCT: lambda x: x,
}
