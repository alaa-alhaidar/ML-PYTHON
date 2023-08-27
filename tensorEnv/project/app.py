import tensorflow as tf
from flask import Flask, render_template, request

app = Flask(__name__)

# Create a global TensorFlow graph instance
global_graph = tf.Graph()
prev_step_output = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global prev_step_output

    if request.method == 'POST':
        step_name = request.form['step_name']
        step_description = request.form['step_description']
        order = int(request.form['order'])
        algorithm_choice = request.form['algorithm_choice']
        metadata = request.form['metadata']

        with global_graph.as_default():
            if prev_step_output is None:
                # If it's the first step, create a constant placeholder
                prev_step_output = tf.compat.v1.placeholder(tf.string, shape=(), name=step_name)

            else:
                # If it's not the first step, link to the previous step using tf.identity()
                prev_step_output = tf.identity(prev_step_output, name=step_name)
            c = tf.add(step_name, prev_step_output, name="c")

        with tf.compat.v1.Session(graph=global_graph) as sess:
            # Perform any necessary TensorFlow operations using sess
            summary_writer = tf.compat.v1.summary.FileWriter("logs/execution_plan")
            summary_writer.add_graph(sess.graph)
            summary_writer.close()

        return render_template('index.html', graph_generated=True)

    return render_template('index.html', graph_generated=False)

if __name__ == '__main__':
    app.run(debug=True)
