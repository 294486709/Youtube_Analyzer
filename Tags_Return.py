#coding=utf8

import random

def tag_return(catogory):
 list = {
    0: "film & comedy",
    1: "music",
    2: "entertainment",
    3: "how to & style",
    4: "science & education",
    5: "sport & gaming"
 }
 if list[catogory] == "film & comedy":
    list0 = {
         0: "Hollywood",
         1: "comedy",
         2: "funny",
         3: "talk show",
         4: "humor",
         5: "joke",
         6: "sketch",
         7: "TV"
      }

    wlatch = []
    num = 0
    wightlist = {0: 8, 1: 37, 2: 44, 3: 8, 4: 11, 5: 15, 6: 15, 7: 7}
    for i in wightlist:
        for x in range(wightlist[i]):
            wlatch.append(i)
            num += 1

    print (wightlist)
    print (num)
    print (wlatch)

    tag_list = []

    while len(tag_list) < 4:
        radnum=random.randint(0,num-1)
        # print(list0[wlatch[radnum]])
        tags = list0[wlatch[radnum]]
        if tags not in tag_list:
            tag_list.append(tags)
    print(tag_list)
    return  tag_list

 if list[catogory] == "music":
    list1 = {
         0: "rock",
         1: "DJ",
         2: "Pop",
         3: "Rap",
         4: "song",
         5: "music video",
         6: "album",
         7: "remix"
      }

    wlatch = []
    num = 0
    wightlist = {0: 3, 1: 6, 2: 22, 3: 5, 4: 15, 5: 8, 6: 5, 7: 3}
    for i in wightlist:
        for x in range(wightlist[i]):
            wlatch.append(i)
            num += 1

    print (wightlist)
    print (num)
    print (wlatch)

    tag_list = []

    while len(tag_list) < 4:
        radnum=random.randint(0,num-1)
        # print(list1[wlatch[radnum]])
        tags = list1[wlatch[radnum]]
        if tags not in tag_list:
            tag_list.append(tags)
    print(tag_list)
    return  tag_list

 if list[catogory] == "entertainment":
    list2 = {
         0: "game",
         1: "show",
         2: "Kardashian",
         3: "celebrity",
         4: "DC",
         5: "interview",
         6: "fashion",
         7: "video"
      }

    wlatch = []
    num = 0
    wightlist = {0: 5, 1: 18, 2: 2, 3: 11, 4: 2, 5: 11, 6: 3, 7: 14}
    for i in wightlist:
        for x in range(wightlist[i]):
            wlatch.append(i)
            num += 1

    print (wightlist)
    print (num)
    print (wlatch)

    tag_list = []

    while len(tag_list) < 4:
        radnum=random.randint(0,num-1)
        # print(list2[wlatch[radnum]])
        tags = list2[wlatch[radnum]]
        if tags not in tag_list:
            tag_list.append(tags)
    print(tag_list)
    return  tag_list

 if list[catogory] == "how to & style":
    list3 = {
         0: "how to",
         1: "tutorial",
         2: "cook",
         3: "style",
         4: "makeup",
         5: "review",
         6: "product",
         7: "guide"
      }

    wlatch = []
    num = 0
    wightlist = {0: 17, 1: 13, 2: 8, 3: 4, 4: 13, 5: 5, 6: 2, 7: 2}
    for i in wightlist:
        for x in range(wightlist[i]):
            wlatch.append(i)
            num += 1

    print (wightlist)
    print (num)
    print (wlatch)

    tag_list = []

    while len(tag_list) < 4:
        radnum=random.randint(0,num-1)
        # print(list3[wlatch[radnum]])
        tags = list3[wlatch[radnum]]
        if tags not in tag_list:
            tag_list.append(tags)
    print(tag_list)
    return  tag_list

 if list[catogory] == "science & education":
    list4 = {
         0: "science",
         1: "TED",
         2: "learn",
         3: "history",
         4: "youtube",
         5: "life",
         6: "space",
         7: "cell"
      }

    wlatch = []
    num = 0
    wightlist = {0: 12, 1: 5, 2: 3, 3: 5, 4: 2, 5: 4, 6: 4, 7: 1}
    for i in wightlist:
        for x in range(wightlist[i]):
            wlatch.append(i)
            num += 1

    print (wightlist)
    print (num)
    print (wlatch)

    tag_list = []

    while len(tag_list) < 4:
        radnum=random.randint(0,num-1)
        # print(list4[wlatch[radnum]])
        tags = list4[wlatch[radnum]]
        if tags not in tag_list:
            tag_list.append(tags)
    print(tag_list)
    return  tag_list

 if list[catogory] == "sport & gaming":
    list5 = {
         0: "NBA",
         1: "football",
         2: "NFL",
         3: "espn",
         4: "wwe",
         5: "red bull",
         6: "product",
         7: "soccer"
      }

    wlatch = []
    num = 0
    wightlist = {0: 6, 1: 5, 2: 2, 3: 2, 4: 2, 5: 1, 6: 2, 7: 3}
    for i in wightlist:
        for x in range(wightlist[i]):
            wlatch.append(i)
            num += 1

    print (wightlist)
    print (num)
    print (wlatch)

    tag_list = []

    while len(tag_list) < 4:
        radnum=random.randint(0,num-1)
        # print(list5[wlatch[radnum]])
        tags = list5[wlatch[radnum]]
        if tags not in tag_list:
            tag_list.append(tags)
    print(tag_list)
    return  tag_list


if __name__ == "__main__":

    tag_return(1)
