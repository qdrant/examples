import { SparseTextEmbedding, SparseEmbeddingModel } from 'fastembed'

export const sparseModel = await SparseTextEmbedding.init({
  model: SparseEmbeddingModel.SpladePPEnV1,
})
