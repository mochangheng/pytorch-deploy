class Logger():
    '''
    Logs information about model progress into a file
    '''
    def __init__(self, filename=None):
        if filename is None:
            self.f = None
        else:
            self.f = open(filename,'a')

    def log(self, message):
        ''' Adds message file '''
        print(message)
        if self.f is not None:
            self.f.write(f'{message}\n')
    
    def close(self):
        ''' Closes the file, instance is invalid after running this ''' 
        if self.f is not None:
            self.f.close()
