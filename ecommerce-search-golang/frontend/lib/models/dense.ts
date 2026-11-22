import { FlagEmbedding, EmbeddingModel } from 'fastembed'

export const denseModel = await FlagEmbedding.init({
  model: EmbeddingModel.BGESmallENV15,
})
