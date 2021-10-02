from tkinter import *
from tkinter import filedialog
#import matplotlib.pyplot as plt


window = Tk()
window.geometry('800x600')
window.title("Построение таблицы связей между элементами на схеме")
lbl = Label(window, text="Подождите, идет подготовка нейросети к работе", font=("Arial Bold", 20))
lbl.grid(column=0, row=0)

import os
import shutil
# файл с именами классов
shutil.copy2("obj.names", "darknet/data/obj.names")
# файл с путями к информации о обучающей и тестовой выборке, именам классов и путем сохранения весов моделей
shutil.copy2("obj.data", "darknet/data/obj.data")
shutil.copy2("yolov4-obj.cfg", "darknet/cfg/yolov4-obj.cfg")

os.system('make')

os.system('''
%cd cfg
!sed -i 's/batch=64/batch=1/' yolov4-obj.cfg
!sed -i 's/subdivisions=32/subdivisions=1/' yolov4-obj.cfg
!sed -i 's/width=640/width=1280/' yolov4-obj.cfg
!sed -i 's/#height=640/#height=1280/' yolov4-obj.cfg
%cd ..
''')


file = filedialog.askopenfilename(title="Выберите файл для распознания")

lbl1 = Label(window, text="Идет обработка изображения", font=("Arial Bold", 20))
lbl1.grid(column=0, row=1)

f = open('images.txt','w')
f.write(file)
f.close()

os.system('./darknet detector test data/obj.data cfg/yolov4-obj.cfg yolov4-obj_last.weights -dont_show -ext_output -thresh 0.1 < /content/images.txt > result.txt')

def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  import matplotlib
  from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
  matplotlib.use('TkAgg')

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  canvas = FigureCanvasTkAgg(fig, master=window)
  plot_widget = canvas.get_tk_widget()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()
  plot_widget.grid(row=2, column=0)

imShow('predictions.jpg')

f = open('classes.txt', 'r')
classes = {}
k=0
for line in f:
  classes[line[:-1]] = k
  k+=1
f.close()
classes

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

f = open('darknet/result.txt', 'r')
k=0
# служебная информация прокрутиться при открытом файле trash
f1 = open('trash.txt', 'w')
for line in f:
  # как только находим название файла, открываем соответствующий ему текстовый файл на запись, закрыв предыдущий
  if line.find('/content/')!= -1:
    f1.close()
    f1 = open('data/scheme_{}.txt'.format(k), 'w')
    k+=1
  # начинаем построчно парсить результаты, записывая их построчно в файл для очередной схемы
  one_obj_list = ''
  string = line.split(' ') # преврящаем очередную строку в список
  # удаляем из списка все пробелы и пустые значения, чтобы очтались только цифры и слова
  string = remove_values_from_list(string, ' ')
  string = remove_values_from_list(string, '')
  if string[0][:-1] in classes.keys(): # если первое значение списка есть в нашем словаре
  # то вычленяем из списка 4 значения для баундинг бокса и приводим их в нужный формат
    one_obj_list = one_obj_list + str(classes[string[0][:-1]]) + ' '
    one_obj_list = one_obj_list + str(int(string[2])/1280 + int(string[6])/1280/2) + ' '
    one_obj_list = one_obj_list + str(int(string[4])/1280 + int(string[8][:-2])/1280/2) + ' '
    one_obj_list = one_obj_list + str(int(string[6])/1280) + ' '
    one_obj_list = one_obj_list + str(int(string[8][:-2])/1280) + '\n'
    f1.write(one_obj_list)
f1.close()

import torchvision.ops.boxes as bops
import torch

def check_intersection(input1, input2):
  xy1 = input1[:2] # Получаем координаты x,y центра 
  wh1 = input1[2:4] # Получаем значения высоты и ширины
  wh_half1 = [x/2. for x in wh1] # Делим значения высоты и ширины пополам
  top_left1 = [x-y for x, y in zip(xy1, wh_half1)] # Получаем значение, соответствующее верхнему левому углу
  right_bottom1 = [x+y for x, y in zip(xy1, wh_half1)] # Получаем значение, соотвествующее правому нижнему углу

  rect1 = [top_left1[0],top_left1[1],right_bottom1[0],right_bottom1[1]]
          
  xy2 = input2[:2] # Получаем координаты x,y центра 
  wh2 = input2[2:4] # Получаем значения высоты и ширины
  wh_half2 = [x/2. for x in wh2] # Делим значения высоты и ширины пополам
  top_left2 = [x-y for x, y in zip(xy2, wh_half2)] # Получаем значение, соответствующее верхнему левому углу
  right_bottom2 = [x+y for x, y in zip(xy2, wh_half2)] # Получаем значение, соотвествующее правому нижнему углу

  rect2 = [top_left2[0],top_left2[1],right_bottom2[0],right_bottom2[1]]

  box1 = torch.tensor([rect1], dtype=torch.float)
  box2 = torch.tensor([rect2], dtype=torch.float)
  iou = bops.box_iou(box1, box2)

  return (float(iou))

# извлекаем из текстового документа данные в массив
f = open('data/scheme_0.txt', 'r')
big_list = []
iou_list = []
for line in f:
  a = line
  my_list = []
  for i in range(5):
    place = a.find(' ')
    a_1 = a[:place]
    a = a[place+1:]
    my_list.append(float(a_1))
  big_list.append(my_list)
# проходимся по всем элементам массива, ищем пересечения
for i in range(len(big_list)):
    for j in range(len(big_list)):
      if check_intersection(big_list[i][1:5],big_list[j][1:5]) !=0 and check_intersection(big_list[i][1:5],big_list[j][1:5]) !=1 and i!=j :
        iou_list.append([i,j])
komponents = [] # заносим в массив все компоненты и точки соединения связей
for i in range(len(big_list)):
  if big_list[i][0]!= 7.0:
    komponents.append(i)
# извлекаем список связей между komponents
fin_links = []
for komponent in komponents:
  links = [komponent]
  done_all = False
  while done_all == False:
    link = komponent
    done = False
    while done == False:
      for i in range(len(iou_list)):
        if iou_list[i][0] == link and (iou_list[i][1] not in links):
          link = iou_list[i][1]
          links.append(link)
          if link in komponents:
            fin_links.append([komponent,link])
            done = True
          break
        elif i == len(iou_list)-1:
          done = True
          done_all = True
# чистим список от повторов
for i in range(len(fin_links)):
  for j in range(len(fin_links)-1):
    if i!=j:
      if fin_links[i][0] == fin_links[j][1] and fin_links[i][1] == fin_links[j][0]:
        fin_links[i] = [-1,-1]
fin_links = list(filter(lambda a: a != [-1,-1], fin_links))
import pandas as pd 

file = filedialog.askopenfilename(title="Выберите папку для сохранения результата")

pd.DataFrame(fin_links).to_csv(file + "fin_links.csv")


window.mainloop()