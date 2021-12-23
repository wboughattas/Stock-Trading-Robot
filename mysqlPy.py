# import mysql
# from mysql.connector import Error
#
#
# def create_server_connection(host_name, user_name, user_password):
#     connection = None
#     try:
#         connection = mysql.connector.connect(
#             host=host_name,
#             user=user_name,
#             passwd=user_password
#         )
#         print("MySQL Server connection successful")
#     except Error as err:
#         print(f"Error: '{err}'")
#
#     return connection
#
#
# def create_db_connection(host_name, user_name, user_password, db_name):
#     connection = None
#     try:
#         connection = mysql.connector.connect(
#             host=host_name,
#             user=user_name,
#             passwd=user_password,
#             database=db_name
#         )
#         print("MySQL Database connection successful")
#     except Error as err:
#         print(f"Error: '{err}'")
#
#     return connection
#
#
# def execute_query(connection, query):
#     cursor = connection.cursor()
#     try:
#         cursor.execute(query)
#         connection.commit()
#         print("Query successful")
#     except Error as err:
#         print(f"Error: '{err}'")
#
#
# def read_query(connection, query):
#     cursor = connection.cursor()
#     result = None
#     try:
#         cursor.execute(query)
#         result = cursor.fetchall()
#         return result
#     except Error as err:
#         print(f"Error: '{err}'")
#
#
# def execute_list_query(connection, sql, val):
#     cursor = connection.cursor()
#     try:
#         cursor.executemany(sql, val)
#         connection.commit()
#         print("Query successful")
#     except Error as err:
#         print(f"Error: '{err}'")
#
#
# # action=['tradebook', 'buy', 'sell', 'quote'], timestamp=['1sec', '1min', '1h', '1d', '1mo']
# def create_ifn(connection, entity, *db_name, action='quote', timestamp='1sec'):
#     cursor = connection.cursor()
#     try:
#         if entity.upper() == 'SCHEMA':
#             cursor.execute(f"CREATE {entity} IF NOT EXISTS {db_name}")
#         elif entity.upper() == 'TABLE':
#             if action != 'quote':
#                 cursor.execute(f"CREATE {entity} IF NOT EXISTS {'-'.join([*db_name, action])}")
#             else:
#                 cursor.execute(f"CREATE {entity} IF NOT EXISTS {'-'.join([*db_name, action, timestamp])}")
#         connection.commit()
#         print("Create_ifn successful")
#     except Error as err:
#         print(f"Error: '{err}'")
