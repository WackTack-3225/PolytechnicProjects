import firebase_admin
from firebase_admin import credentials, db

def initialize_firebase():
    """Initializes Firebase application with given credentials and database URL."""
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://your_google_firebase.com/"
    })

def update_employee_data(employee_data):
    """Updates employee data in the Firebase Realtime Database."""
    ref = db.reference('Employee')
    for key, value in employee_data.items():
        ref.child(key).set(value)
        print(f"Updated data for {value['Name']}")

if __name__ == "__main__":
    # Employee data to be updated in Firebase
    data = {
        "ID": {
            "Name": "John Doe",
            "Employee ID": "ID",
            "Position": "Worker"
        }
    }
    # To add data, follow the format below
    # data = {
    #     "Worker 1 ID": {
    #         "Name": "Worker 1 Name",
    #         "Employee ID": "Worker 1 ID",
    #         "Position": "Worker 1 Position"
    #     },
    #     "Worker 2 ID": {
    #             "Name": "Worker 2 Name",
    #             "Employee ID": "Worker 2 ID",
    #             "Position": "Worker 2 Position"
    #         },
    # }

    # Initialize Firebase and update employee data
    initialize_firebase()
    update_employee_data(data)