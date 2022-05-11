from bertopic import BERTopic

class BertTopicForSummarization():

    def __init__(self, verbose: bool,
                 min_topic_size: int):
        self.topic_model = BERTopic(verbose=verbose,
                                    embedding_model="paraphrase-MiniLM-L12-v2",
                                    min_topic_size=min_topic_size)

    def __call__(self, article_batches: str):
        topics, _ = self.topic_model.fit_transform(article_batches); len(self.topic_model.get_topic_info())
        print(topics)

