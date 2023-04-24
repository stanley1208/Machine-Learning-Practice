import time

def runtime():
    def now_time():
        print(time.time())
    return now_time

f=runtime()
f()

def make_filter(keep):
    def the_filter(file_name):
        file=open(file_name)
        lines=file.readlines()
        file.close()
        filter_doc=[i for i in lines if keep in i]
        return filter_doc
    return the_filter

filter1=make_filter("8")
filter_result=filter1("data.csv")
print(filter_result)