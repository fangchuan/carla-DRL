import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

from functools import wraps
import datetime
import traceback

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def email_sender(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        ret = True

        # 第三方 SMTP 服务
        mail_sender = 'mf1623010@smail.nju.edu.cn'  # 发件人邮箱账号
        sender_passwd = 'FangChuan99'  # 发件人邮箱密码
        mail_receiver = '1457737815@qq.com'  # 收件人邮箱账号，我这边发送给自己

        server = smtplib.SMTP_SSL("smtp.exmail.qq.com", 465)  # 发件人邮箱中的SMTP服务器，端口是465

        # 标准邮件需要三个头部信息： From, To, 和 Subject
        start_time = datetime.datetime.now()
        start_msg = 'Your training has started. \n' \
                    'Main call: {func_name} \n' \
                    'Starting date: {date}'.format(func_name=func.__doc__,
                                                   date=start_time.strftime(DATE_FORMAT))
        start_message = MIMEText(start_msg, 'plain', 'utf-8')
        start_message['From'] = formataddr(["fc-128gSSD", mail_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
        start_message['To'] = formataddr(["fc", mail_receiver])  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
        start_message['Subject'] = "训练开始邮件"  # 邮件的主题，也可以说是标题

        try:
            server.login(mail_sender, sender_passwd)  # 括号中对应的是发件人邮箱账号、邮箱密码
            server.sendmail(mail_sender, [mail_receiver, ], start_message.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
        except Exception:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
            ret = False
            return ret

        finish_msg = func.__doc__
        try:
            func(*args, **kwargs)
            end_time = datetime.datetime.now()
            elaspe_time = end_time - start_time
            finish_msg =  'Your training has finished normally.\n' \
                            'Main call:{func_name}\n'  \
                            'Finish date: {date}\n'  \
                            'duration:{duration} '.format(func_name=func.__doc__,
                                                            date=end_time.strftime(DATE_FORMAT),
                                                            duration=str(elaspe_time))
        except BaseException as be:
            end_time = datetime.datetime.now()
            elapsed_time = end_time - start_time
            finish_msg = "Your training has crashed.\n" \
                        'Main call: {func_name}\n'  \
                        'Starting date: {start_date}\n' \
                        'Crash date: {crash_date}\n' \
                        'Crashed training duration: {duration}\n' \
                        "Here's the error:" \
                        '{exception}\n'  \
                        "Traceback:\n" \
                        '{traceback}'.format(func_name=func.__doc__,
                                             start_date=start_time.strftime(DATE_FORMAT),
                                             crash_date=end_time.strftime(DATE_FORMAT),
                                             duration= str(elapsed_time),
                                             exception=be,
                                             traceback=traceback.format_exc())

        finish_message = MIMEText(finish_msg, 'plain', 'utf-8')
        finish_message['From'] = formataddr(["fc-128gSSD", mail_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
        finish_message['To'] = formataddr(["fc", mail_receiver])  # 括号里的对应收件人邮箱昵称、收件人邮箱账号
        finish_message['Subject'] = "训练结束邮件"  # 邮件的主题，也可以说是标题

        try:
            server.login(mail_sender, sender_passwd)  # 括号中对应的是发件人邮箱账号、邮箱密码
            server.sendmail(mail_sender, [mail_receiver, ], finish_message.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
            server.quit()
        except Exception:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
            ret = False
            return ret

        return ret

    return wrapper


# test the email_sender()
if __name__ == "__main__":

    import time
    @email_sender
    def test():
        '''
        fuck you, it's only a test function
        '''
        print('fuck you and you and you')
        time.sleep(10)

        return True

    if test():
        print("Send mail successfully.")
    else:
        print("Error: Cannot send email.")
