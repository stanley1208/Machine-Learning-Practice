import time

def runtime(func):
    def get_time():
        print(time.time())
        func()
    return get_time

@runtime
def student_run():
    print("run")

student_run()

# parameters decorator
def runtime(func):
    def get_time(*args):
        print(time.time())
        func(*args)
    return get_time


@runtime
def student_run(i,j):
    print("run")

@runtime
def student_run1(i):
    print("run")

student_run(1,2)
student_run1(1)

# key parameters
def runtime(func):
    def get_time(*args,**kwargs):
        print(time.time())
        func(*args,**kwargs)
    return get_time


@runtime
def student_run(i,j):
    print("run")

@runtime
def student_run1(*args,**kwargs):
    print("run")

@runtime
def student_run2():
    print("run")

student_run(1,2)
student_run1(5,i=1,j=2)
student_run2()



