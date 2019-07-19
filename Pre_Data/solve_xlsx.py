import xlrd

book = xlrd.open_workbook('cmy.xlsx')

for sheet in book.sheets():
    print sheet.name

sheet = book.sheet_by_name('Sheet1')

print sheet.nrows

BsmI = []
AapI = []
FokI = []
TaqI = []
Cdx2 = []


for i in range(sheet.nrows):
    print sheet.row_values(i)
    if 'BsmI' in sheet.row_values(i)[1]:
        BsmI.append( sheet.row_values(i)[0])
    if 'AapI' in sheet.row_values(i)[1]:
        AapI.append( sheet.row_values(i)[0])
    if 'FokI' in sheet.row_values(i)[1]:
        FokI.append( sheet.row_values(i)[0])
    if 'TaqI' in sheet.row_values(i)[1]:
        TaqI.append( sheet.row_values(i)[0])
    if 'Cdx2' in sheet.row_values(i)[1]:
        Cdx2.append( sheet.row_values(i)[0])

print 'BsmI'
print BsmI
print '=' * 20

print 'AapI'
print AapI
print '=' * 20

print 'FokI'
print FokI
print '=' * 20

print 'TaqI'
print TaqI
print '=' * 20

print 'Cdx2'
print Cdx2
print '=' * 20


