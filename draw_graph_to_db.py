import pymysql
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)
plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
mydb = pymysql.connect(host = 'localhost', user = 'root', password = '1111', database = 'test_pymysql', charset = 'utf8', autocommit = True)
mycursor = mydb.cursor()

mycursor.execute("DROP TABLE Region_Data")
mycursor.execute("CREATE TABLE Region_Data(ID INT(30), Time_In_Region INT(30), From_Where VARCHAR(255))")
# 저장된 database를 통해서 region에 대한 분석
in_region_num = mycursor.execute("SELECT ID From People_Detection WHERE State = 'In_The_Smoke'")
in_region_data = mycursor.fetchall()
for num in range(in_region_num):
    tuple_num = mycursor.execute("SELECT * From People_Detection WHERE ID = %s", (in_region_data[num][0]))
    tuple_data = mycursor.fetchall()
    region_time = tuple_data[tuple_num - 1][4]
    if(region_time >= 300):
        region_id = tuple_data[tuple_num - 1][0]
        region_state_num = mycursor.execute("SELECT * From People_Detection WHERE State != ' ' and ID = %s", (in_region_data[num][0]))
        region_state_data = mycursor.fetchall()
        for num2 in range(region_state_num):
            if(region_state_data[num2][3] == 'Out_From_One'):
                region_state = 'Out_From_One'
                break
            elif(region_state_data[num2][3] == 'Out_From_New'):
                region_state = 'Out_From_New'
                break
            elif(region_state_data[num2][3] == 'Out_From_Lib'):
                region_state = 'Out_From_Lib'
                break
            else:
                region_state = 'Not_Found'
        mycursor.execute("INSERT INTO Region_Data Values (%s, %s, %s)", (region_id, region_time, region_state))

#원흥관 IN/OUT
print('--------------')
print('원흥관 IN/OUT')
one_in = []
one_out = []
one_total = mycursor.execute("SELECT ID, State, OUT_ONE, IN_ONE, Seconds FROM counting WHERE State = 'Out_From_One' or State = 'In_To_One'")
one_total_data = mycursor.fetchall()
last_seconds = one_total_data[one_total - 1][4]
count = (last_seconds // 60) + 1
start_time = 1
end_time = count
for i in range(count):
    if i == 0:
        total = mycursor.execute("SELECT ID, State, OUT_ONE, IN_ONE, Seconds FROM counting WHERE (Seconds < %s and Seconds >= %s) and (State = 'Out_From_One' or State = 'In_To_One')", ((i + 1) * 60, i * 60))
        one_in.insert(i, one_total_data[total - 1][3])
        one_out.insert(i, one_total_data[total - 1][2])
        print('%d 분에서 %d 분까지 이동->'%(i, i + 1), 'Out: ', one_out[i], 'In: ', one_in[i])
    else:
        one_in_last = one_total_data[total - 1][3]
        one_out_last = one_total_data[total - 1][2]
        total = total + mycursor.execute("SELECT ID, State, OUT_ONE, IN_ONE, Seconds FROM counting WHERE (Seconds < %s and Seconds >= %s) and (State = 'Out_From_One' or State = 'In_To_One')", ((i + 1) * 60, i * 60))
        one_in.insert(i, one_total_data[total - 1][3] - one_in_last)
        one_out.insert(i, one_total_data[total - 1][2] - one_out_last)
        print('%d 분에서 %d 분까지 이동->'%(i, i + 1), 'Out: ', one_out[i], 'In: ', one_in[i])

x = np.arange(start_time, end_time + 1, 1)
y1 = np.array(one_out)
y2 = np.array(one_in)
ax1.set_title('One IN/OUT')
ax1.set_xlabel('TIME(minute)')
ax1.set_ylabel('COUNTS(in/out)')
ax1.plot(x, y1, color = 'b', linestyle = '-', marker = 'o', label = 'OUT')
ax1.plot(x, y2, color = 'r', linestyle = '--', marker = 'v', label = 'IN')
ax1.legend(loc = 'best')

#신공학관 IN/OUT
print('신공학관 IN/OUT')
new_in = []
new_out = []
new_total = mycursor.execute("SELECT ID, State, OUT_NEW, IN_NEW, Seconds FROM counting WHERE State = 'Out_From_New' or State = 'In_To_New'")
new_total_data = mycursor.fetchall()
last_seconds = new_total_data[new_total - 1][4]
count = (last_seconds // 60) + 1
start_time = 1
end_time = count
for i in range(count):
    if i == 0:
        total = mycursor.execute("SELECT ID, State, OUT_NEW, IN_NEW, Seconds FROM counting WHERE (Seconds < %s and Seconds >= %s) and (State = 'Out_From_New' or State = 'In_To_New')", ((i + 1) * 60, i * 60))
        new_in.insert(i, new_total_data[total - 1][3])
        new_out.insert(i, new_total_data[total - 1][2])
        print('%d 분에서 %d 분까지 이동->'%(i, i + 1), 'Out: ', new_out[i], 'In: ', new_in[i])
    else:
        new_in_last = new_total_data[total - 1][3]
        new_out_last = new_total_data[total - 1][2]
        total = total + mycursor.execute("SELECT ID, State, OUT_NEW, IN_NEW, Seconds FROM counting WHERE (Seconds < %s and Seconds >= %s) and (State = 'Out_From_New' or State = 'In_To_New')", ((i + 1) * 60, i * 60))
        new_in.insert(i, new_total_data[total - 1][3] - new_in_last)
        new_out.insert(i, new_total_data[total - 1][2] - new_out_last)
        print('%d 분에서 %d 분까지 이동->'%(i, i + 1), 'Out: ', new_out[i], 'In: ', new_in[i])

x = np.arange(start_time, end_time + 1, 1)
y1 = np.array(new_out)
y2 = np.array(new_in)
ax2.set_title('New IN/OUT')
ax2.set_xlabel('TIME(minute)')
ax2.set_ylabel('COUNTS(in/out)')
ax2.plot(x, y1, color = 'b', linestyle = '-', marker = 'o', label = 'OUT')
ax2.plot(x, y2, color = 'r', linestyle = '--', marker = 'v', label = 'IN')
ax2.legend(loc = 'best')

#도서관 IN/OUT
print('--------------')
print('도서관 IN/OUT')
lib_in = []
lib_out = []
lib_total = mycursor.execute("SELECT ID, State, OUT_LIB, IN_LIB, Seconds FROM counting WHERE State = 'Out_From_Lib' or State = 'In_To_Lib'")
lib_total_data = mycursor.fetchall()
last_seconds = lib_total_data[lib_total - 1][4]
count = (last_seconds // 60) + 1
start_time = 1
end_time = count
for i in range(count):
    if i == 0:
        total = mycursor.execute("SELECT ID, State, OUT_LIB, IN_LIB, Seconds FROM counting WHERE (Seconds < %s and Seconds >= %s) and (State = 'Out_From_Lib' or State = 'In_To_Lib')", ((i + 1) * 60, i * 60))
        lib_in.insert(i, lib_total_data[total - 1][3])
        lib_out.insert(i, lib_total_data[total - 1][2])
        print('%d 분에서 %d 분까지 이동->'%(i, i + 1), 'Out: ', lib_out[i], 'In: ', lib_in[i])
    else:
        lib_in_last = lib_total_data[total - 1][3]
        lib_out_last = lib_total_data[total - 1][2]
        total = total + mycursor.execute("SELECT ID, State, OUT_LIB, IN_LIB, Seconds FROM counting WHERE (Seconds < %s and Seconds >= %s) and (State = 'Out_From_Lib' or State = 'In_To_Lib')", ((i + 1) * 60, i * 60))
        lib_in.insert(i, lib_total_data[total - 1][3] - lib_in_last)
        lib_out.insert(i, lib_total_data[total - 1][2] - lib_out_last)
        print('%d 분에서 %d 분까지 이동->'%(i, i + 1), 'Out: ', lib_out[i], 'In: ', lib_in[i])

x = np.arange(start_time, end_time + 1, 1)
y1 = np.array(lib_out)
y2 = np.array(lib_in)
ax3.set_title('Lib IN/OUT')
ax3.set_xlabel('TIME(minute)')
ax3.set_ylabel('COUNTS(in/out)')
ax3.plot(x, y1, color = 'b', linestyle = '-', marker = 'o', label = 'OUT')
ax3.plot(x, y2, color = 'r', linestyle = '--', marker = 'v', label = 'IN')
ax3.legend(loc = 'best')

#region 시간 분석
region_num = mycursor.execute("SELECT * FROM region_data")
region_dataset = mycursor.fetchall()
not_found_time = 0
one_out_time = 0
lib_out_time = 0
new_out_time = 0
for num in range(region_num):
    if region_dataset[num][2] == 'Not_Found':
        not_found_time += region_dataset[num][1]
    elif region_dataset[num][2] == 'Out_From_One':
        one_out_time += region_dataset[num][1]
    elif region_dataset[num][2] == 'Out_From_New':
        new_out_time += region_dataset[num][1]
    elif region_dataset[num][2] == 'Out_From_In':
        lib_out_time += region_dataset[num][1]
region_time = []
region_label = ['Not_Found', 'Out_From_One', 'Out_From_New', 'Out_From_Lib']
x = np.array(region_label)
not_found_time = not_found_time // 30
one_out_time = one_out_time // 30
lib_out_time = lib_out_time // 30
new_out_time = new_out_time // 30
region_time.insert(0, not_found_time)
region_time.insert(1, one_out_time)
region_time.insert(2, new_out_time)
region_time.insert(3, lib_out_time)
y = np.array(region_time)
n_groups = len(x)
index = np.arange(n_groups)
ax4.set_title('Region Analysis')
ax4.set_ylabel('Time(seconds)')
ax4.set_xlabel('Where')
ax4.bar(index, y, tick_label = x, align = 'center')

plt.grid(True)
plt.show()