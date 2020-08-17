class Logger():
    '''
    Logs information about model progress into a file
    '''
    def __init__(self, filename):
        self.f = open(filename,'a')

    def log(self, message):
        ''' Adds message file '''
        self.f.write(f'{message}\n')
    
    def close(self)
        ''' Closes the file, instance is invalid after running this ''' 
        self.f.close()
