import socket
import sys
import struct
import time

class point:
    def __init__(self):
        self.valueX = 0.0
        self.valueY = 0.0
        self.valueZ = 0.0
        self.valueA = 0.0
        self.valueB = 0.0
        self.valueC = 0.0
        self.valueTool = False

    def setValues(self, list):
        self.valueX = list[0]
        self.valueY = list[1]
        self.valueZ = list[2]
        self.valueA = list[3]
        self.valueB = list[4]
        self.valueC = list[5]
        self.valueV = list[6]
        self.valueTool = list[7]

    def toByteArray(self):
        ba = bytearray(struct.pack("f", self.valueX)) + bytearray(struct.pack("f", self.valueY)) + bytearray(
            struct.pack("f", self.valueZ)) + bytearray(struct.pack("f", self.valueA)) + bytearray(
            struct.pack("f", self.valueB)) + bytearray(struct.pack("f", self.valueC)) + bytearray(struct.pack("f", self.valueV)) + bytearray(
            struct.pack("?", self.valueTool))
        s = " END"
        ba.extend(s.encode())

        return ba


class Server:
    iPAdress = '192.168.1.151'
    port = 1999
    connection = []
    client_address = []
    p1 = point()
    def __init__(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        server_address = (self.iPAdress, self.port)
        print('starting up on %s port %s' % server_address, file=sys.stderr)
        sock.bind(server_address)
        sock.listen(5)
        # print ('waiting for a connection', file=sys.stderr)
        # connection, client_address = sock.accept()
        print('waiting for a connection', file=sys.stderr)
        self.connection, self.client_address = sock.accept()
        print('connection from', self.client_address, file=sys.stderr)

    def setTimeout(self, value):
        return time.time() + value

    def TransmissionClose(self):
        self.connection.close()


    def printData(self, value):
        res = []
        for i in range(6):
            arr = bytearray([value[i * 4], value[i * 4 + 1], value[i * 4 + 2], value[i * 4 + 3]])
            res.append(struct.unpack('f', arr)[0])

        if bytearray(bytearray(value[6 * 4]) == b''):
            print("ziemniaki")
            res.append("False")
        else:
            res.append("True")

        return res

    def Transmission(self, list):
        self.p1.setValues([list[0], list[1], list[2], 75, -85, -165, list[4], list[3]])

        while True:
            print("Start sending point")
            try:

                timeout = self.setTimeout(10)
                while True:

                    if time.time() > timeout:
                        break

                    self.connection.sendall(self.p1.toByteArray())

                    data = self.connection.recv(33)
                    data = bytearray(data)
                    if (data):
                        print(self.printData(data))
                        return 1
            finally:
                print("done")

