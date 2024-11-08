

def load_raw_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data

def load_question_answer(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        qa_data = file.read()
        qa_texts = [f"Question: {qa['question']} Answer: {qa['answer']}" for qa in qa_data]
    return data

def load_chunk(file_path,chunk_size):
    with open(file_path, 'r', encoding='utf-8') as file:
        chunk = file.read(chunk_size)
    return chunk
