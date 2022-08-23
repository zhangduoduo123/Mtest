from django.db import models

# Create your models here.

class Res(models.Model):
    naifenname = models.CharField(max_length=255, blank=True, null=True)
    distancename = models.CharField(max_length=255)
    mama1 = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'res'



class File(models.Model):
    filename = models.CharField(max_length=255, blank=True, null=True)
    upload_time = models.CharField(max_length=255, blank=True, null=True)
    filepath = models.CharField(max_length=255, blank=True, null=True)
    userid = models.ForeignKey('User', models.DO_NOTHING, db_column='userid', blank=True, null=True)
    active = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'file'
    def __unicode__(self):
        return self.id


class User(models.Model):
    userid = models.AutoField(primary_key=True)
    username = models.CharField(max_length=255, blank=True, null=True)
    passward = models.CharField(max_length=255, blank=True, null=True)
    email = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'user'
    def __unicode__(self):
        return self.userid

class Xiangdui(models.Model):
    mother_id = models.IntegerField()
    city = models.IntegerField(blank=True, null=True)
    mother_name = models.CharField(max_length=255, blank=True, null=True)
    nfamily = models.IntegerField(blank=True, null=True)
    nchild = models.IntegerField(blank=True, null=True)
    minzu = models.IntegerField(blank=True, null=True)
    age1 = models.IntegerField(blank=True, null=True)
    age = models.IntegerField(blank=True, null=True)
    chanci = models.IntegerField(blank=True, null=True)
    tall = models.IntegerField(blank=True, null=True)
    weight1 = models.FloatField(blank=True, null=True)
    weight2 = models.FloatField(blank=True, null=True)
    weight3 = models.FloatField(blank=True, null=True)
    fenmian_way = models.IntegerField(blank=True, null=True)
    birthday = models.DateField(blank=True, null=True)
    birth_jd = models.IntegerField(blank=True, null=True)
    birthweight = models.FloatField(blank=True, null=True)
    birthtall = models.FloatField(blank=True, null=True)
    sex = models.CharField(max_length=255, blank=True, null=True)
    birth_suit = models.IntegerField(blank=True, null=True)
    yangben_id = models.CharField(max_length=255, blank=True, null=True)
    pro = models.FloatField(blank=True, null=True)
    day = models.CharField(max_length=255, blank=True, null=True)
    miruqi = models.IntegerField(blank=True, null=True)
    caiyang_time = models.IntegerField(blank=True, null=True)
    lai = models.FloatField(blank=True, null=True)
    su = models.FloatField(blank=True, null=True)
    jie = models.FloatField(blank=True, null=True)
    dan = models.FloatField(blank=True, null=True)
    yiliang = models.FloatField(blank=True, null=True)
    liang = models.FloatField(blank=True, null=True)
    benbing = models.FloatField(blank=True, null=True)
    se = models.FloatField(blank=True, null=True)
    zu = models.FloatField(blank=True, null=True)
    tiandong = models.FloatField(blank=True, null=True)
    si = models.FloatField(blank=True, null=True)
    gu = models.FloatField(blank=True, null=True)
    gan = models.FloatField(blank=True, null=True)
    bing = models.FloatField(blank=True, null=True)
    lao = models.FloatField(blank=True, null=True)
    jing = models.FloatField(blank=True, null=True)
    fu = models.FloatField(blank=True, null=True)
    banguang = models.FloatField(blank=True, null=True)
    sumaa = models.FloatField(blank=True, null=True)
    baa = models.FloatField(blank=True, null=True)
    fbaa = models.FloatField(blank=True, null=True)
    b2fb = models.FloatField(blank=True, null=True)
    b2sum = models.FloatField(blank=True, null=True)
    fb2sum = models.FloatField(blank=True, null=True)
    insertpeople = models.ForeignKey(User, models.DO_NOTHING, db_column='insertpeople', blank=True, null=True)

    class Meta:
            managed = False
            db_table = 'xiangdui'


class Yuanshi(models.Model):
    mother_id = models.IntegerField()
    city = models.IntegerField(blank=True, null=True)
    mother_name = models.CharField(max_length=255, blank=True, null=True)
    nfamily = models.IntegerField(blank=True, null=True)
    nchild = models.IntegerField(blank=True, null=True)
    minzu = models.IntegerField(blank=True, null=True)
    age = models.IntegerField(blank=True, null=True)
    age1 = models.IntegerField(blank=True, null=True)
    chanci = models.IntegerField(blank=True, null=True)
    tall = models.IntegerField(blank=True, null=True)
    weight1 = models.FloatField(blank=True, null=True)
    weight2 = models.FloatField(blank=True, null=True)
    weight3 = models.FloatField(blank=True, null=True)
    fenmian_way = models.IntegerField(blank=True, null=True)
    birthday = models.DateField(blank=True, null=True)
    birth_jd = models.IntegerField(blank=True, null=True)
    birthweight = models.FloatField(blank=True, null=True)
    birthtall = models.FloatField(blank=True, null=True)
    sex = models.CharField(max_length=255, blank=True, null=True)
    birth_suit = models.IntegerField(blank=True, null=True)
    yangben_id = models.CharField(max_length=255, blank=True, null=True)
    pro = models.FloatField(blank=True, null=True)
    day = models.CharField(max_length=255, blank=True, null=True)
    miruqi = models.IntegerField(blank=True, null=True)
    caiyang_time = models.IntegerField(blank=True, null=True)
    lai = models.FloatField(blank=True, null=True)
    su = models.FloatField(blank=True, null=True)
    jie = models.FloatField(blank=True, null=True)
    dan = models.FloatField(blank=True, null=True)
    yiliang = models.FloatField(blank=True, null=True)
    liang = models.FloatField(blank=True, null=True)
    benbing = models.FloatField(blank=True, null=True)
    se = models.FloatField(blank=True, null=True)
    zu = models.FloatField(blank=True, null=True)
    tiandong = models.FloatField(blank=True, null=True)
    si = models.FloatField(blank=True, null=True)
    gu = models.FloatField(blank=True, null=True)
    gan = models.FloatField(blank=True, null=True)
    bing = models.FloatField(blank=True, null=True)
    lao = models.FloatField(blank=True, null=True)
    jing = models.FloatField(blank=True, null=True)
    fu = models.FloatField(blank=True, null=True)
    banguang = models.FloatField(blank=True, null=True)
    sumaa = models.FloatField(blank=True, null=True)
    baa = models.FloatField(blank=True, null=True)
    fbaa = models.FloatField(blank=True, null=True)
    b2fb = models.FloatField(blank=True, null=True)
    b2sum = models.FloatField(blank=True, null=True)
    fb2sum = models.FloatField(blank=True, null=True)
    insertpeople = models.ForeignKey(User, models.DO_NOTHING, db_column='insertpeople', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'yuanshi'

class NfXiangdui(models.Model):
    insertpeople = models.ForeignKey('User', models.DO_NOTHING, db_column='insertpeople', blank=True, null=True)
    nf_name = models.CharField(primary_key=True, max_length=255)
    lai = models.FloatField(blank=True, null=True)
    su = models.FloatField(blank=True, null=True)
    jie = models.FloatField(blank=True, null=True)
    dan = models.FloatField(blank=True, null=True)
    yiliang = models.FloatField(blank=True, null=True)
    liang = models.FloatField(blank=True, null=True)
    benbing = models.FloatField(blank=True, null=True)
    se = models.FloatField(blank=True, null=True)
    zu = models.FloatField(blank=True, null=True)
    tiandong = models.FloatField(blank=True, null=True)
    si = models.FloatField(blank=True, null=True)
    gu = models.FloatField(blank=True, null=True)
    gan = models.FloatField(blank=True, null=True)
    bing = models.FloatField(blank=True, null=True)
    lao = models.FloatField(blank=True, null=True)
    jing = models.FloatField(blank=True, null=True)
    fu = models.FloatField(blank=True, null=True)
    banguang = models.FloatField(blank=True, null=True)
    baa = models.FloatField(blank=True, null=True)
    fbaa = models.FloatField(blank=True, null=True)
    b2fb = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'nf_xiangdui'


class NfYuanshi(models.Model):
    nf_name = models.CharField(primary_key=True, max_length=255)
    pro = models.FloatField(blank=True, null=True)
    lai = models.FloatField(blank=True, null=True)
    su = models.FloatField(blank=True, null=True)
    jie = models.FloatField(blank=True, null=True)
    dan = models.FloatField(blank=True, null=True)
    yiliang = models.FloatField(blank=True, null=True)
    liang = models.FloatField(blank=True, null=True)
    benbing = models.FloatField(blank=True, null=True)
    se = models.FloatField(blank=True, null=True)
    zu = models.FloatField(blank=True, null=True)
    tiandong = models.FloatField(blank=True, null=True)
    si = models.FloatField(blank=True, null=True)
    gu = models.FloatField(blank=True, null=True)
    gan = models.FloatField(blank=True, null=True)
    bing = models.FloatField(blank=True, null=True)
    lao = models.FloatField(blank=True, null=True)
    jing = models.FloatField(blank=True, null=True)
    fu = models.FloatField(blank=True, null=True)
    banguang = models.FloatField(blank=True, null=True)
    sumaa = models.FloatField(blank=True, null=True)
    baa = models.FloatField(blank=True, null=True)
    fbaa = models.FloatField(blank=True, null=True)
    b2sum = models.FloatField(blank=True, null=True)
    b2fb = models.FloatField(blank=True, null=True)
    fb2sum = models.FloatField(blank=True, null=True)
    insertpeople = models.ForeignKey('User', models.DO_NOTHING, db_column='insertpeople', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'nf_yuanshi'


class Protain(models.Model):
    insert_people = models.ForeignKey('User', models.DO_NOTHING, db_column='insert_people', blank=True, null=True)
    mother_id = models.IntegerField(blank=True, null=True)
    city = models.IntegerField(blank=True, null=True)
    mother_name = models.CharField(max_length=255, blank=True, null=True)
    nfamily = models.IntegerField(db_column='NFAMILY', blank=True, null=True)  # Field name made lowercase.
    nchild = models.IntegerField(db_column='NCHILD', blank=True, null=True)  # Field name made lowercase.
    minzu = models.IntegerField(blank=True, null=True)
    age = models.IntegerField(blank=True, null=True)
    age1 = models.IntegerField(blank=True, null=True)
    chanci = models.IntegerField(blank=True, null=True)
    tall = models.IntegerField(blank=True, null=True)
    weight1 = models.FloatField(blank=True, null=True)
    weight2 = models.FloatField(blank=True, null=True)
    weight3 = models.FloatField(blank=True, null=True)
    fenmian_way = models.IntegerField(blank=True, null=True)
    birthday = models.DateField(blank=True, null=True)
    birth_jd = models.IntegerField(blank=True, null=True)
    birthweight = models.IntegerField(blank=True, null=True)
    birthtall = models.IntegerField(blank=True, null=True)
    sex = models.IntegerField(blank=True, null=True)
    birth_suit = models.IntegerField(blank=True, null=True)
    sample_id = models.CharField(max_length=255, blank=True, null=True)
    times = models.IntegerField(blank=True, null=True)
    miruqi = models.IntegerField(blank=True, null=True)
    caiyang_day = models.CharField(max_length=255, blank=True, null=True)
    α_rubai = models.FloatField(blank=True, null=True)
    β_lao = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'protain'



class Tempyuanshi(models.Model):
    lai = models.FloatField(blank=True, null=True)
    su = models.FloatField(blank=True, null=True)
    jie = models.FloatField(blank=True, null=True)
    dan = models.FloatField(blank=True, null=True)
    yiliang = models.FloatField(blank=True, null=True)
    liang = models.FloatField(blank=True, null=True)
    benbing = models.FloatField(blank=True, null=True)
    se = models.FloatField(blank=True, null=True)
    zu = models.FloatField(blank=True, null=True)
    tiandong = models.FloatField(blank=True, null=True)
    si = models.FloatField(blank=True, null=True)
    gu = models.FloatField(blank=True, null=True)
    gan = models.FloatField(blank=True, null=True)
    bing = models.FloatField(blank=True, null=True)
    lao = models.FloatField(blank=True, null=True)
    jing = models.FloatField(blank=True, null=True)
    fu = models.FloatField(blank=True, null=True)
    banguang = models.FloatField(blank=True, null=True)
    sumaa = models.FloatField(blank=True, null=True)
    baa = models.FloatField(blank=True, null=True)
    fbaa = models.FloatField(blank=True, null=True)
    b2fb = models.FloatField(blank=True, null=True)
    b2sum = models.FloatField(blank=True, null=True)
    fb2sum = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'tempyuanshi'



class Tempxiangdui(models.Model):
    lai = models.FloatField(blank=True, null=True)
    su = models.FloatField(blank=True, null=True)
    jie = models.FloatField(blank=True, null=True)
    dan = models.FloatField(blank=True, null=True)
    yiliang = models.FloatField(blank=True, null=True)
    liang = models.FloatField(blank=True, null=True)
    benbing = models.FloatField(blank=True, null=True)
    se = models.FloatField(blank=True, null=True)
    zu = models.FloatField(blank=True, null=True)
    tiandong = models.FloatField(blank=True, null=True)
    si = models.FloatField(blank=True, null=True)
    gu = models.FloatField(blank=True, null=True)
    gan = models.FloatField(blank=True, null=True)
    bing = models.FloatField(blank=True, null=True)
    lao = models.FloatField(blank=True, null=True)
    jing = models.FloatField(blank=True, null=True)
    fu = models.FloatField(blank=True, null=True)
    banguang = models.FloatField(blank=True, null=True)
    sumaa = models.FloatField(blank=True, null=True)
    baa = models.FloatField(blank=True, null=True)
    fbaa = models.FloatField(blank=True, null=True)
    b2fb = models.FloatField(blank=True, null=True)
    b2sum = models.FloatField(blank=True, null=True)
    fb2sum = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'tempxiangdui'
