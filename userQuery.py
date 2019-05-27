#Runhao Zhao (rz6dg) Wenxi Zhao (wz8nx) Shaoran Li (sl4bz) Winfred Hill (whh3rz) Jingnan Yang (jy4fch)
#import the data set
import pandas as pd

# Print the table when the input is "1"
def userInput1(dataset):
    top20Rating = dataset[["RATING","NAME"]]
    top20Rating = top20Rating.sort_values(["RATING"],ascending=False)[:20]
    
    download = input("Do you want to output it as csv? y/n ")
    if download == "y":
        top20Rating.to_csv("Top20Rating.csv", encoding="latin-1",index=False)
        print(top20Rating)
        
    quited = input("Do you want to quit? y/n")
    if quited == "n":
        return True
    if quited =="y": 
        return False
    
# Print the table when the input is 2   
def userInput2(dataset):
    top20Countries = pd.DataFrame(dataset["Nationality"].value_counts()[:20])
    top20Countries.reset_index(level=0, inplace=True)
    top20Countries.columns =["Nationality","Frequency"]
        
    download = input("Do you want to output it as csv? y/n ")
    if download == "y":
        top20Countries.to_csv("Top20Countries.csv", encoding="latin-1",index=False)
    print(top20Countries)
        
    quited = input("Do you want to quit? y/n")
    if quited == "n":
        return True
    if quited =="y": 
        return False

# Print table when the input is "3"    
def userInput3(dataset):
    top20Wage = dataset[["WAGE","NAME"]]
    top20Wage = top20Wage.sort_values(["WAGE"],ascending=False)[:20]
    download = input("Do you want to output it as csv? y/n ")
        
    if download == "y":
        top20Wage.to_csv("Top20Wage.csv",encoding="latin-1",index=False)
    print(top20Wage)
        
    quited = input("Do you want to quit? y/n")
    if quited == "n":
        return True
    if quited =="y": 
        return False
 
# Print table when the input is "4"       
def userInput4(dataset):
    top50Defender = dataset[["Defending","NAME"]]
    top50Defender = top50Defender.sort_values(["Defending"],ascending=False)[:50]
        
    download = input("Do you want to output it as csv? y/n ")
    if download =="y":
        top50Defender.to_csv("Top20Defender.csv",encoding="latin-1",index=False)
    print(top50Defender)
        
    quited = input("Do you want to quit? y/n")
    if quited == "n":
        return True
    if quited =="y": 
        return False

# Main method of the userQuery
def userQuery():
    # Import the data set
    dataset = pd.read_csv("dataForAnalysis2.csv", encoding="latin-1")
    
    # Start the loop to prompt the user to choose commands
    nobreak = True
    while nobreak:
        print("1. top20 Players with highest rating\n2. top20 countries have most soccer players\n3. top20 Players with highest Wage\n4. top50 players with highest defending score")
        #ask for user input
        userInput = input("Please enter a number to select the data set: ")
        #output the dataset needed
        if userInput == "1":
            nobreak = userInput1(dataset)
        elif userInput == "2":
            nobreak = userInput2(dataset)
        elif userInput == "3":
            nobreak = userInput3(dataset)
        elif userInput == "4":
            nobreak = userInput4(dataset)
        else:
            print("Please enter a right number! ")

userQuery() # Call the userQuery