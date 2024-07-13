# Intoduction
This folder is used for my work done in my tenure as the Head of Secretary. I implemeted some automation techniques for my club via Apps Script powered by Google.

Mainly the codes here are for an attendance/database management system in Google Sheets. 

From a data engineering standpoint, Apps script is only good for low latency processes and really does not handle large data inputs well in any shape or form. This is not for an industrial application or for a large company in any shape or form, it is for Secretaries or roles with the relevant responsibilties to manage a club of >50-100 students.

The code is stored in a text file, copy paste it into the apps script extenion provided by google as a new project. 

# Projects

* Admin System
> The admin system only checks the admin number(A Unique Identifier) to determine if someone exists in the database. The name of the admin number field or any fied in particular is not important BUT the column in which the admin numbers are located is important as the code takes the data starting from `Row:2, Column:3`. The way to use them is technically inside so I will not explain. 

* Subcom Attendance System
> Same as the admin system, it only checks the position of the data, not the name of the column. So feel free to replace the Admin Number with any header you like. 

# How to use this:
1) Download the excel file, save to your google drive. 
    - **Important: DELETE ALL FIELDS FOR DEPLOYMENT, THE PROVIDED FIELDS ARE FOR TESTING HOW IT WORKS**
2) Copy paste the code into apps script. Give your script a name
3) Click the run button in apps script to see if it works, if it does just refresh the Google Sheets Page and you should see the `Student Management` Tab (or a new tab in general) at the end of the other tabs in google sheets (Its in the image provided in the excel file)
    - Please make sure that you check the indentations as its not formatted properly in .txt if im not wrong 
4) *(Optional-Subcommitee Attendance System)* You can replace all fields that are true false with checkboxes. It looks nicer.

# Conclusions:
As I have finised my tenure already, I wont look into improving or updating any of these as really, I will probably never use these ever again. I hope that this does improve your Quality of life as it did for mine. Cheers and Happy Managing!