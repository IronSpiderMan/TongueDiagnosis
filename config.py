class Config(object):
    TESTING = False


class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = 'fasdfdsafjs8932rjfewfs'
    DATABASE_URI = 'mysql://user@localhost/foo'


class DevelopmentConfig(Config):
    SECRET_KEY = 'fasdfdsafjs8932rjfewfs'
    DATABASE_URI = "sqlite:////tmp/foo.db"


class TestingConfig(Config):
    DATABASE_URI = 'sqlite:///:memory:'
    TESTING = True
