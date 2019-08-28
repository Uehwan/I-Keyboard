import glob
import pickle
import os


data_path = "../raw_data/*.txt"


def find_files(path): return glob.glob(path)


def make_data_from_text(path):
    char, x_data, y_data = [], [], []
    temp = 0

    f = open(path, 'r')
    lines = f.readlines()
    previous_check = '1'
    for line in lines:
        check = list(line)[0]
        # for the first sentence
        if not check.isdigit() and previous_check.isdigit():
            current_char = []
            for element in list(line):
                current_char.append(element)
            if '\r' in current_char:
                current_char.remove('\r')
            if '\n' in current_char:
                current_char.remove('\n')
            char.append(current_char)
            temp = len(current_char)
            flag = 0

        # for the second sentence if it has
        if not check.isdigit() and not previous_check.isdigit():
            char.pop()
            current_char.append('\n')
            temp = len(current_char)
            flag = 1
            char.append(current_char)
            current_char = []
            for element in list(line)[:-1]:
                current_char.append(element)
            char.append(current_char)

        # x_coordinate touch
        if check.isdigit() and not previous_check.isdigit():
            x = line.split(',')
            current_x = []
            for element in x:
                current_x.append(int(element))
            if flag == 1:
                x1, x2 = current_x[:temp], current_x[temp:]
                x_data.append(x1)
                x_data.append(x2)
            else:
                x_data.append(current_x)

        # y_coordinate touch
        if previous_check.isdigit() and check.isdigit():
            y = line.split(',')
            current_y = []
            for element in y:
                current_y.append(int(element))
            if flag == 1:
                y1, y2 = current_y[:temp], current_y[temp:]
                y_data.append(y1)
                y_data.append(y2)
            else:
                y_data.append(current_y)

        # first element of the previous line
        previous_check = list(line)[0]

    f.close()

    word_data = []
    for i in range(len(char)):
        current_char = char[i]
        single_sentence = []
        current_word = ''
        previous_word = ''

        for element in current_char:
            if element != ' ' and element != '.' and element!='\n':
                current_word += element
                single_sentence.append(previous_word)
            elif element == '.':
                single_sentence.append(current_word)
            elif element == '\n':
                single_sentence.append(current_word)
            else:
                single_sentence.append(current_word)
                previous_word = current_word
                current_word = ''
        word_data.append(single_sentence)

    return char, x_data, y_data, word_data


# use function make_data_from_text
def data_generation(path):
    char, x_data, y_data, word_data = [], [], [], []

    for filename in find_files(path):
        char_one, x_one , y_one, word_one = make_data_from_text(filename)
        for sublist in char_one:
            char.append(sublist)
        for sublist in x_one:
            x_data.append(sublist)
        for sublist in y_one:
            y_data.append(sublist)
        for sublist in word_one:
            word_data.append(sublist)
    return char, x_data, y_data, word_data


if __name__ == "__main__":
    o_count = 0
    x_count = 0
    print("data generation starts...")

    list_of_dirs = ["list_data/o/char/", "list_data/o/x_data/", "list_data/o/y_data/",
                    "list_data/x/char/", "list_data/x/x_data/", "list_data/x/y_data/"]
    for dirs in list_of_dirs:
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    for filename in find_files(data_path):
        print("Current file:", filename)
        char, x_data, y_data, word_data = data_generation(filename)

        if filename[-5] == "o":
            with open("list_data/o/char/char"+str(o_count)+".txt", "wb") as fp:
                pickle.dump(char, fp)

            with open("list_data/o/x_data/x_data"+str(o_count)+".txt", "wb") as fp:
                pickle.dump(x_data, fp)

            with open("list_data/o/y_data/y_data"+str(o_count)+".txt", "wb") as fp:
                pickle.dump(y_data, fp)
            o_count += 1
            print("o_count:", o_count)
        elif filename[-5] == "x":
            with open("list_data/x/char/char" + str(x_count) + ".txt", "wb") as fp:
                pickle.dump(char, fp)

            with open("list_data/x/x_data/x_data" + str(x_count) + ".txt", "wb") as fp:
                pickle.dump(x_data, fp)

            with open("list_data/x/y_data/y_data" + str(x_count) + ".txt", "wb") as fp:
                pickle.dump(y_data, fp)
            x_count += 1
            print("x_count:", x_count)
