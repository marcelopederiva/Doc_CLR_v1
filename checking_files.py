def filter_vehicle_car(input_text):
    lines = input_text.strip().split("\n\n")
    filtered_lines = []
    # print(len(lines))
    # exit()
    for line in lines:
        if line.startswith("vehicle.car"):
            parts = line.split()
            if float(parts[3])> 0:
                label = parts[0]
                values = [f"{float(x):.2f}" for x in parts[1:]]
                filtered_line = f"{label} {' '.join(values)}"
                filtered_lines.append(filtered_line)
    
    return "\n".join(filtered_lines)

def read_and_filter(input_path, output_path):
    with open(input_path, 'r') as file:
        input_text = file.read()
        
    filtered_text = filter_vehicle_car(input_text)
    
    with open(output_path, 'w') as file:
        # file.write("Filtered Array:\\n")
        file.write(filtered_text)
        # file.write("\\n")

if __name__ == "__main__":
    input_path = "arrays/"
    name= '9ca6b6b0daa048c4a5322ff944f193dc'
    # name = '983eed02192d46d5b4df7abb905155f5'
    # name = 'f0cde9ca2b0c48c0935d2a23290fc4b0'
    # name = 'f9bdc7dd40074505bf81c3eef5f13dca'
    output_path = "arrays/filtered_"+name+'.txt'
    read_and_filter(input_path+name+'.txt', output_path)