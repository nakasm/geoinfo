'''
Created on 2022/03/11

@author: sin
'''

class Bghash:
    '''
    binary geohash
    '''
    MAX_LAT = 90.0
    MIN_LAT = -90.0
    MAX_LONG = 180.0
    MIN_LONG = -180.0
    
    GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    GEOHASH_CHARVAL = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, \
                       '8': 8, '9': 9, 'b': 10, 'c': 11, 'd': 12, 'e': 13, 'f': 14, 'g': 15, \
                       'h': 16, 'j': 17, 'k': 18, 'm': 19, 'n': 20, 'p': 21, 'q': 22, 'r': 23, 's': 24, \
                       't': 25, 'u': 26, 'v': 27, 'w': 28, 'x': 29, 'y': 30, 'z': 31}
    DEFAULT_PRECISION = 41
    MAXIMUM_PRECISION = 58
    
    NORTH = 0
    NORTHEAST = 1
    EAST = 2
    SOUTHEAST = 3
    SOUTH = 4
    SOUTHWEST = 5
    WEST = 6
    NORTHWEST = 7
    
    def __init__(self, lat_or_hash = None, lon_or_prec = None, precision = None) :
        self.code = 0
        if precision is None:
            if isinstance(lat_or_hash, int) and lon_or_prec is None :
                '''
                from a single int (64bit) or a hex string representing a bghash
                '''
                self.code = lat_or_hash & 0xffffffffffffffff
                return
            elif isinstance(lat_or_hash, str) and lon_or_prec is None :
                try:
                    int_value = int(lat_or_hash)
                    self.code = int_value & 0xffffffffffffffff
                    return
                except ValueError:
                    self.degeohash(lat_or_hash)
                    return
            elif lat_or_hash is not None and lon_or_prec is not None : 
                precision = min(int(lon_or_prec), self.MAXIMUM_PRECISION)
                if isinstance(lat_or_hash, str):
                    self.set(int(lat_or_hash), precision)
                else:
                    self.set(lat_or_hash, precision)
                return
        elif isinstance(lat_or_hash, float) and isinstance(lon_or_prec, float):
            latitude = lat_or_hash
            longitude = lon_or_prec
            if precision is None :
                precision = self.DEFAULT_PRECISION
            else:
                precision = min(int(precision), self.MAXIMUM_PRECISION)

            if -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0 :
                northsouth = [self.MAX_LAT, self.MIN_LAT]
                eastwest   = [self.MAX_LONG, self.MIN_LONG]
                bit = 1 <<63
                i = 0
                while i < precision:
                    latmid = (northsouth[0] + northsouth[1]) / 2.0
                    lonmid = (eastwest[0] + eastwest[1]) / 2.0
                    if longitude > lonmid :
                        eastwest[1] = lonmid
                        self.code |= bit
                    else:
                        eastwest[0] = lonmid                
                    bit >>= 1
                    i += 1
                    if not i < precision :
                        break
                    if latitude > latmid :
                        northsouth[1] = latmid
                        self.code |= bit
                    else: 
                        northsouth[0] = latmid
                    bit >>= 1
                    i += 1
            self.set_precision(precision)
            return

    def precision(self):
        return self.code & 0x3f

    def binarycode(self):
        return self.code  & (0xffffffffffffffff ^ 0x3f)

    def set_precision(self, prec):
        self.code &= (0xffffffffffffffff << (64 - prec))
        self.code |= (prec & 0x3f)
        return self

    def set(self, value, prec = None):
        if prec is None :
            prec = value & 0x3f
        if prec > 58 :
            prec = 58
        self.code = value & (0xffffffffffffffff << (64 - prec))
        self.code |= prec
        return self
      
    def __str__(self):
        return hex(self.code)
    
    def __lt__(self, another):
        if self.precision() == another.precision() :
            return self.code < another.code
        minprec = min(self.precision(), another.precision())
        mask = 0xffffffffffffffff << (64 - minprec)
        return (self.code & mask) < (another.code & mask)

    def degeohash(self, codestr):
        binvalue = 0
        precision = len(codestr)*5
        for ch in codestr:
            binvalue <<= 5
            binvalue |= self.GEOHASH_CHARVAL[ch]
        binvalue <<= (64 - precision)
        self.set(binvalue, precision)
        return 
    
    def decode(self):
        northsouth = [self.MAX_LAT, self.MIN_LAT]
        eastwest   = [self.MAX_LONG, self.MIN_LONG]

        if self.code > 0 :
            checkbit = 1 <<63
            i = 0
            while i < self.precision() :
                delta = (eastwest[0] - eastwest[1]) / 2.0
                if (self.code & checkbit) != 0 :
                    eastwest[1] += delta
                else:
                    eastwest[0] -= delta
                checkbit >>= 1
                i += 1
                if not i < self.precision():
                    break
                delta = (northsouth[0] - northsouth[1]) / 2.0
                if (self.code & checkbit) != 0 :
                    northsouth[1] += delta
                else:
                    northsouth[0] -= delta
                checkbit >>= 1
                i += 1
        return {'n': northsouth[0], 's': northsouth[1], 'e': eastwest[0], 'w': eastwest[1]}
    
    def neighbor(self, d):
        bcode=self.binarycode()
        prec = self.precision()
       # print(d,bcode,prec)
        lsb = 1<<(64 - prec)
        lon = bcode & 0xaaaaaaaaaaaaaaaa
        lat = bcode & 0x5555555555555555
        if d == 'n' or d == 'ne' or d == 'nw' or d == 0 or d == 1 or d == 7 :
            lat = ((lat | 0xaaaaaaaaaaaaaaaa) + lsb) & 0x5555555555555555
        elif d == 's' or d == 'se' or d == 'sw' or d == 4 or d == 3 or d == 5 :
            lat = (lat - lsb) & 0x5555555555555555
        if d == 'e' or d == 'ne' or d == 'se' or d == 2 or d == 1 or d == 3 :
            lon = ((lon | 0x5555555555555555) + lsb) & 0xaaaaaaaaaaaaaaaa
        elif d == 'w' or d == 'sw' or d == 'nw' or d == 6 or d == 5 or d == 7 :
            lon = (lon - lsb) & 0xaaaaaaaaaaaaaaaa

        return Bghash( lon | lat | prec )
        

    def geohash(self, length=None):
        ghash = ''
        if length is None :
            length = int(self.precision() / 5)
        for pos in range(0, length) :
            ghash += Bghash.GEOHASH_BASE32[self.code>>(59 - pos * 5) & 0x1f]
        return ghash
    