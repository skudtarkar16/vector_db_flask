import re
from flask import Flask, request, jsonify
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

app = Flask(__name__)
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
print("\nLoaded the model\n")
vector_db_path = "faiss_collection/faiss_e5_base"
faiss_db = FAISS.load_local(vector_db_path, embeddings)


def get_similar_query_faiss(query):
    output = faiss_db.similarity_search_with_score(query, k=3)
    question_list = []
    for op in output:
        output_chunk = op[0].page_content
        output_chunk = output_chunk.replace("\n\n", "\n")
        similarity_score = op[1]
        match_ = re.search(r'Questions:\n(.+)', output_chunk)

        scrapped_query = match_.group(1)

        question_list.append((scrapped_query, similarity_score))
    top_match = question_list[0][0]
    top_match_score = question_list[0][1]
    # return question_list
    return top_match, top_match_score


@app.route('/', methods=['POST'])
def main_flask_fn():
    req_as_json = request.get_json(silent=True, force=True)
    # print(req_as_json)
    userquery = req_as_json.get('userquery')
    answer, score = get_similar_query_faiss(userquery)
    score = float(score)
    res = jsonify({'matchedquery': answer, 'matchscore': score})
    return res


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
