#coding=utf-8 
#!/usr/bin/python
import pymysql    #导入模块
# import yaml

# Configure db
# conf = yaml.load(open('mysql_demo.yaml'))

# mysql = pymysql.connect(host=conf['mysql_host'], user=conf['mysql_user'], password=conf['mysql_password'], database =conf['mysql_db'], charset=conf['mysql_charset'])  
# host=localhost #也可以写,如果127.0.0.1不能用的话#  登录数据库
# cur = conn.cursor()   # 数据库操作符 游标
class MysqlUtil():
    def __init__(self,host='rm-2zef251c5ev7a9tkh.mysql.rds.aliyuncs.com',db='tianping',user='tianping_api',passwd='Rtmap911',port=3306,charset='utf8'):
        self.host=host
        self.port=port
        self.db=db
        self.user=user
        self.passwd=passwd
        self.charset=charset

    def connect(self):
        self.conn=pymysql.connect(host=self.host,
                        port=self.port,
                        db=self.db,
                        user=self.user,
                        passwd=self.passwd)
        self.cursor=self.conn.cursor(cursor=pymysql.cursors.DictCursor)

    def close(self):
        '''关闭数据库连接'''
        self.cursor.close()
        self.conn.close()

    def get(self,sql):
        try:
            self.connect()
            '''数据库单条查询'''
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            self.close()
            return results
        except Exception as e:
            return e.args

    def crud(self,sql,params):
        try:
            # print(params)
            self.connect()
            rows = self.cursor.execute(sql,params)
            self.conn.commit()
            self.close()
            return rows
        except Exception as e:
            print(e.args)
            return e.args

    def crudAndId(self, sql, params):
        try:
            # print(params)
            self.connect()
            self.cursor.execute(sql,params)
            lastrowid = self.cursor.lastrowid
            self.conn.commit()
            self.close()
            return lastrowid
        except Exception as e:
            return e.args

    def rows(self,sql):
        try:
            self.connect()
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            self.close()
            return result
        except Exception as e:
            return e.args
