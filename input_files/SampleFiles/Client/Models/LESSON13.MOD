MFWMOD12
~Description
Lesson 13 - Working with Multiple Instances of a Report
Use this model to begin Lesson 13.
~Settings
M/D/Y
50
.
0
1
1
,
0
N
N
N
Y
~ExtendedSettings
4
N
null
1
1
~TableField
Media
4
0
0
1
C
0
5
4
N
N
4
5
0
0
Y
~TableFieldExtended
Media
3
N
0
0
~TableField
Description
39
0
0
1
C
0
39
6
N
N
39
5
0
11
N
~TableFieldExtended
Description
3
N
0
0
~TableField
Label Number
12
0
0
1
C
0
12
7
N
N
12
5
0
52
N
~TableFieldExtended
Label Number
3
N
0
0
~TableField
Quantity
4
0
0
1
N
0
8
5
N
N
4
5
0
5
N
~TableFieldExtended
Quantity
3
N
0
0
~TableField
Amount
8
2
0
1
N
0
6
9
N
N
8
5
0
73
N
~TableFieldExtended
Amount
3
N
0
0
~TableField
Customer
25
0
0
1
C
0
25
2
N
N
25
3
0
10
N
~TableFieldExtended
Customer
3
N
0
0
~TableField
Unit Price
6
2
0
1
N
0
10
8
N
N
6
5
0
65
N
~TableFieldExtended
Unit Price
3
N
0
0
~TableField
Ship Date
8
0
0
1
D
0
10
3
N
N
8
1
0
50
N
~TableFieldExtended
Ship Date
3
N
1
1
~TableField
Report Date
8
0
0
1
D
0
11
1
N
N
8
0
0
0
N
~TableFieldExtended
Report Date
3
N
1
1
~Level
0
ÑÑ/
1
05/01/98                  CLASSICAL MUSIC DISTRIBUTORS                   PAGE 01
1
0
N
0
0
~LevelExtended
0
1
Page Header
N
~Level
1
                          :                     :
1
              ORDER NUMBER:  536017    SHIP DATE: 04/06/98
1
0
N
0
0
~LevelExtended
1
1
Order Number Level
N
~Level
3
CUSTOMER:
4
CUSTOMER: Betty's Music Store
          Muscatine Plaza
          200 Lower Muscatine
          Cedar Falls, IA 50613
1
0
N
10
25
~LevelExtended
3
1
Customer Level
N
~Level
5
        ÑßßØ
1
CD      4  Bartok, Sonata for Solo Violin           MK-42625       8.99    35.96
7
0
N
0
0
~LevelExtended
5
1
Line Item Detail
N
~Filter
bluegrass_
NBluegrass Records
Customer="Bluegrass*"
~Filter2
bluegrass_
2
Y
0
~Filter
big_shanty
NBig Shanty Music
Customer="Big Shanty*"
~Filter2
big_shanty
2
Y
0
~Summary
SALES_BY_M
N
2
2
Customer
N
0
26
0
Report Date
Y
0
8
0
1
Amount
11
~Summary2Info
SALES_BY_M
2
Customer
N
N
N
N
Report Date
N
N
N
N
0
Y
Bar (Vertical)
0
N
N
N
N
N
N
N
5
Y



~Summary3Info
SALES_BY_M
Y
2
1
Customer
Subtotal
Report Date
Subtotal
~Summary4Info
SALES_BY_M
10
Sales By Month
N
N

N
N
1
1
1
0
13
2
Customer

N
0
N
All Others
0
N
0
1
0

0
Report Date

N
0
N
All Others
0
N
0
1
0

0
5
0
5
1
0 Amount

11
2

~PageSetup
1
256
10
50
50
2
Y
Y
Y

~PageSetupExtended
1
Y
~End
