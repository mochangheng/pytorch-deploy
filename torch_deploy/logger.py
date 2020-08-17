class Logger():
    def __init__(self, filename):
        self.f = open(filename,'a')

    def log(self, message):
        self.f.write(f'{message}\n')
    
    def close(self)
        self.f.close()