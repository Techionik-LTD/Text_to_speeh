from flask import Flask, request, redirect, url_for, flash, render_template, session
import os
import pyttsx3
import random
import speech_recognition as sr
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
app.secret_key = 'supersecretkey'

ALLOWED_EXTENSIONS = {'txt'}

# Initialize model and tokenizer at the start
model = None
tokenizer = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def train_gpt2_model(file_path):
    global model, tokenizer
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Create a dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Train the model
    trainer.train()

def generate_response_gpt2(model, tokenizer, user_input, max_length=50, used_responses=set()):
    inputs = tokenizer.encode(user_input, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        top_k=50,  # top-k sampling
        top_p=0.95,  # nucleus sampling
        temperature=0.7,  # control randomness
        repetition_penalty=2.0  # penalize repetition
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Avoid repeating responses
    attempt_count = 0
    while response in used_responses and attempt_count < 5:
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=2.0
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        attempt_count += 1

    if response not in used_responses:
        used_responses.add(response)

    return response

def introduce():
    greetings = [
        "Meu nome é Gold Rogers, presto muitos serviços, fique à vontade para me perguntar. Estou aqui para lhe dar respostas com perguntas apropriadas",
        "Meu nome é Gold Rogers, presto muitos serviços, fique à vontade para me perguntar. Estou aqui para lhe dar respostas com perguntas apropriadas",
        "Meu nome é Gold Rogers, presto muitos serviços, fique à vontade para me perguntar. Estou aqui para lhe dar respostas com perguntas apropriadas"
    ]
    return random.choice(greetings)

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Ouvindo...")
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio, language='pt-BR')
            print(f"Usuário: {user_input}")
            return user_input
        except sr.UnknownValueError:
            print("Desculpe, não consegui entender o áudio.")
            return None
        except sr.RequestError:
            print("Não foi possível solicitar resultados; verifique sua conexão de rede.")
            return None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Train the model on the uploaded file
            train_gpt2_model(file_path)
            flash('Agent ready for conversation')
            return redirect(url_for('select_voice'))
    return render_template('upload.html')
@app.route('/choose_voice', methods=['GET', 'POST'])
def select_voice():
    if request.method == 'POST':
        voice_type = int(request.form['voice_type'])
        session['voice_type'] = voice_type
        return redirect(url_for('chat'))

    return render_template('choose_voice.html')


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global model, tokenizer
    tts_engine = pyttsx3.init()
    used_responses = set()
    greeting_given = False

    voice_type = int(session.get('voice_type', 0))
    voices = tts_engine.getProperty('voices')
    if voice_type == 1:
        tts_engine.setProperty('voice', voices[1].id)
    else:
        tts_engine.setProperty('voice', voices[0].id)

    tts_engine.setProperty('rate', 180)

    if request.method == 'POST':
        if not greeting_given:
            greeting = introduce()
            print(f"AI: {greeting}")
            tts_engine.say(greeting)
            tts_engine.runAndWait()
            greeting_given = True

        user_input = recognize_speech()
        if user_input is None:
            flash('Could not understand the audio')
            return redirect(request.url)

        if user_input.lower() in ["sair", "parar", "exit", "quit"]:
            flash('Session ended')
            return redirect(url_for('upload_file'))

        response = generate_response_gpt2(model, tokenizer, user_input, used_responses=used_responses)
        print(f"AI: {response}")
        tts_engine.say(response)
        tts_engine.runAndWait()

    return render_template('chat.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
