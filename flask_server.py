from flask import Flask, request, jsonify
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)

# MySQL 연결 설정
def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="1022",
            database="test_db"
        )
        print("MySQL Database connection successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection

# 데이터 삽입 API
@app.route('/insert_violation', methods=['POST'])
def insert_violation():
    data = request.get_json()
    license_plate = data.get("license_plate")
    image_path = data.get("image_path")

    connection = create_connection()
    cursor = connection.cursor()
    try:
        query = "INSERT INTO violations (license_plate, image_path) VALUES (%s, %s)"
        cursor.execute(query, (license_plate, image_path))
        connection.commit()
        return jsonify({"message": "Violation inserted successfully"}), 200
    except Error as e:
        return jsonify({"error": str(e)}), 500

# 데이터 조회 API
@app.route('/get_violations', methods=['GET'])
def get_violations():
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        query = "SELECT * FROM violations;"
        cursor.execute(query)
        violations = cursor.fetchall()
        return jsonify(violations), 200
    except Error as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
