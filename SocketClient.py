import socket


host = '127.0.0.1'
port = 12000
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host,port))

while True:
    data = sock.recv(1024)
    if data == None or data == (b''):
        print('You Fucked Up.')
        break
    elif(data == (b'Cut')):
        print('Received : ',data)
        sock.close()
        exit(1)
    elif(data == (b'True')):
        print('Received : ',data)
        print('Flag : True ')
    elif(data == 'False'):
        print('Received : ',data)
        print('Flag : False ')
        

  
sock.close()


