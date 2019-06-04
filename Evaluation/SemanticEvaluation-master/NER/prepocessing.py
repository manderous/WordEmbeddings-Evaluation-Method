
with open("./data/CoNLL-2003/eng.testb","r") as readFile:
    list=readFile.readlines()
with open("./data/scnu_CoNLL2003/eng.testb.txt","w") as writeFile:
    for i in list:
        if(i!='\n'):
            newlist=i.split(' ')
            writeFile.write(str(newlist[0]+' '+str(newlist[3])))
        else:writeFile.write('\n')