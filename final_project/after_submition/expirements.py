import subprocess

# Set up hyper-parameters
optimizers = ['SGD', 'Adam', 'AdamW']
learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
momentums = [0.1, 0.5, 0.9]
data = "/home/liranc6/Using_YOLOv8/taco_trash_dataset.yaml"
prefix_command = f"yolo task=detect mode=train model=yolov8s.pt imgsz=640 data={data} epochs=10 workers=2"
suffix_command = "2>&1 | tee "

# Run each combination for 10 epochs and collect results
for optimizer in optimizers:
    for learning_rate in learning_rates:
        for momentum in momentums:
            # Run the command to train the model with the current hyperparameters
            name = f"optimizer={str(optimizer)}_learning_rate={str(learning_rate)}_momentum={str(momentum)}"  # result folder name
            output_file = name.replace('.', '_').replace('=', '_') + ".txt"
            command = "srun -c 2 --gres=gpu:1 {} optimizer={} lrf={} momentum={} name={} {} {}" \
                      "".format(prefix_command, str(optimizer), str(learning_rate), str(momentum), name, suffix_command, output_file)
            print("running now: \n{}".format(command))  # follow progress
            ###run
            output = subprocess.check_output(command, shell=True)
            print(output.decode())
            print("finished running command")