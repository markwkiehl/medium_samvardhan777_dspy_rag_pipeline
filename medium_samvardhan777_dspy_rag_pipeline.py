#
#   Mark W Kiehl
#   http://mechatronicsolutionsllc.com/
#   http://www.savvysolutions.info/savvycodesolutions/
#

if __name__ == '__main__':
	openai_api_key = "update with your OpenAPI key"
	
	
    # Load an online article into a Qdrant vector database and then one-shot query with DSPy
    # source: https://medium.com/@samvardhan777/exploring-the-dspy-framework-building-simple-rag-pipelines-2efa0efa634b
    # https://github.com/samvardhan777/DSPy_Playground/blob/main/dspy_qdrant.ipynb

    # Extract the contents of an online article using LlamaIndex.
    # LlamaIndex (GPT Index) is a data framework for your LLM application.
    # https://github.com/qdrant/qdrant-client
    # https://docs.llamaindex.ai/en/stable/examples/data_connectors/WebPageDemo/
    # pip install llama-index llama-index-readers-web
    from llama_index.core import SummaryIndex
    from llama_index.readers.web import SimpleWebPageReader
    url = "https://www.thoughtworks.com/en-in/insights/blog/data-strategy/building-an-amazon-com-for-your-data-products"
    documents = SimpleWebPageReader(html_to_text=True).load_data(
        [url]
    )
    doc_contents = [doc.text for doc in documents]      # This content is too long and causes the model content length to be 4097 tokens, exceeding the max allowed.
    # Define a new, smaller doc_contents ..
    doc_contents = ["Thoughtworks believes data products should be discoverable, addressable, trustworthy, self-describing, interoperable and secure. I", " Thoughtworks, for example, often identifies potential data products by working backwards from the use case using the Jobs to be done (JTBD) framework created by Clayton Christensen. ", "The two best ways to fail at creating valuable, reusable data products are   to develop them without any sense of who they are for and to make them more complicated than they need to be. ", "A key part of data product thinking is keeping the consumers at the center and considering what provides the most value for them. The only way to ensure we are delivering high-quality data products is to identify those consumers, understand their requirements and codify their expectations within a SLO/SLI framework. "]
    doc_ids = list(range(1, len(doc_contents) + 1))
    #print("\ndoc_contents:", doc_contents)
    print("doc_ids:", doc_ids)  # doc_ids: [1, 2, 3, 4]
    doc_metadata = [
        {"source": "Thoughtworks"},
        {"source": url},
    ]

    # Qdrant vector search engine.
    # https://github.com/qdrant/qdrant-client
    # pip install qdrant-client
    # pip install fastembed
    from qdrant_client import QdrantClient
    client = QdrantClient(":memory:")

    client.add(
        collection_name="DSpy_Qdrant",
        documents=doc_contents,
        metadata=doc_metadata,
        ids=doc_ids,
    )

    # View the document..
    #search_result = client.query(collection_name="DSpy_Qdrant",query_text="This is a query document")
    #print(search_result)

    # https://dspy-docs.vercel.app/deep-dive/retrieval_models_clients/QdrantRM/
    from dspy.retrieve.qdrant_rm import QdrantRM
    import dspy

    retriever_model = QdrantRM("DSpy_Qdrant", client, k=3)
    #  k (int, optional): The default number of top passages to retrieve. Default: 3.

    # Define and configure the LLM
    # model="mistral" is obsolete / depreciated.  "gpt-3.5-turbo-instruct" is a suitable substitute.  See: https://platform.openai.com/docs/models/model-endpoint-compatibility
    llm = dspy.LM(model="gpt-3.5-turbo-instruct",
                            model_type='text',
                            #max_tokens=350,        # max_tokens is for setting the size reservation of the response.
                            #temperature=0.1,
                            #top_p=0.8, frequency_penalty=1.17, top_k=40, 
                            api_key=openai_api_key)

    dspy.settings.configure(lm=llm, rm=retriever_model)


    class GenerateAnswer(dspy.Signature):
        """Answer questions with short factoid answers."""

        context = dspy.InputField(desc="may contain relevant facts")
        question = dspy.InputField()
        answer = dspy.OutputField()

    class RAG(dspy.Module):
        def __init__(self, num_passages=3):
            super().__init__()

            self.retrieve = dspy.Retrieve(k=num_passages)
            self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        
        def forward(self, question):
            context = self.retrieve(question).passages
            prediction = self.generate_answer(context=context, question=question)
            return dspy.Prediction(context=context, answer=prediction.answer)
    

    #print("GenerateAnswer.signature:", GenerateAnswer.signature)        # GenerateAnswer.signature: context, question -> answer
    #print("GenerateAnswer.instructions:", GenerateAnswer.instructions)      # GenerateAnswer.instructions: Answer questions with short factoid answers.

    from dspy.signatures import signature_to_template
    template = signature_to_template(GenerateAnswer)
    #print("template:", str(template))           # template: Template(Answer questions with short factoid answers., ['Context:', 'Question:', 'Answer:'])

    qa = RAG()

    pred = qa("How does Thoughtworks identify potential data products?")
    # AssertionError: No RM is loaded.

    print(f"Predicted Answer: {pred.answer}")
    # Predicted Answer: Jobs to be done (JTBD) framework created by Clayton Christensen.
