import subprocess
import time

def get_cpu_temp():
    while True:
        result = subprocess.run(['sensors'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        inside_coretemp_block = False

        for line in result.stdout.split('\n'):
            if 'coretemp-isa-0000' in line:
                inside_coretemp_block = True
            elif inside_coretemp_block and line.strip() == '':
                # Stop when you reach an empty line (indicating the next block)
                break
            if inside_coretemp_block:
                print(line)

        time.sleep(1)  # Sleep for 1 second before checking again

# Run the function
get_cpu_temp()