from getpass import getpass
import requests

def get_token():
    username = input("Enter your username: ")
    password = getpass("Enter your password: ")

    payload = {'grant_type': '', 
          'username': username, 
          'password': password, 
          'scope': '', 
          'client_id': '', 
          'client_secret': ''}
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    token = requests.post("http://127.0.0.1:8000/token", data=payload, headers=headers)
    return token.json()

if __name__ == '__main__':
    print(get_token())
