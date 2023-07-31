from flask import Flask, render_template
import model
import random

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    model_filename, unique_notes, note_to_int, int_to_note = model.run_training()
    seed_notes = random.choices(unique_notes, k=40)  # Randomly select seed notes
    generated_melody = model.load_and_generate(model_filename, unique_notes, note_to_int, int_to_note, seed_notes)
    midi_path = 'generated_song.mid'
    generated_melody.write('midi', midi_path)
    return midi_path

if __name__ == '__main__':
    app.run(debug=True)
