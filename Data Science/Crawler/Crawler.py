import requests
import re
import sys
import time
from bs4 import BeautifulSoup
import multiprocessing
from multiprocessing import Pool

ptt_URL = 'https://www.ptt.cc'
sys.setrecursionlimit(10000)
months = [1, 2, 3, 4, 5, 6 ,7 ,8, 9, 10, 11, 12]
days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
process_num = multiprocessing.cpu_count()

def crawl_(req_page):
	time.sleep(0.1)
	r = requests.get(req_page)
	content = r.text
	return content
#==================for crawl func.========================================
def parse_article_info(content):
	soup = BeautifulSoup(content, 'html.parser')
	all_article = soup.find_all('div', class_= 'r-ent')
	info = []
	for aa in all_article:
		if aa.find('a'):
			tmp = {
				'title': aa.find('div', class_= 'title').text.strip(),
				'push': aa.find('div', class_= 'nrec').text,
				'date': aa.find('div', class_= 'date').text,
				'link_URL': ptt_URL+aa.find('a')['href'],
				}
			info.append(tmp)
	return info
#==================for crawl func.========================================
def parse_contents_info(article):

	cont = crawl_(article['link_URL'])
	soup = BeautifulSoup(cont, 'html.parser')
	time = ''	
	if '1/01' in article['date'] or '12/31' in article['date']:
		metaline_list = soup.find_all('div', class_= 'article-metaline')
		for meta in metaline_list:
			if meta.find('span', 'article-meta-tag').text == '時間':
				time = meta.find('span', 'article-meta-value').text
		if time[-4:] != '2017':
			return '' 
	f2_tag = soup.find_all('span', class_='f2')					
	for f in f2_tag:
		if "發信站" in f.text:
			#store realted information to the list
			output_str = str(article['date'] + ',' + article['title'] + ',' + article['link_URL'])
			if '爆' in article['push']:
				return "P"+output_str
			else:
				return "N"+output_str
	return ''
#==================for push func.========================================
def parse_push_info(content):

	p_list = []
	soup = BeautifulSoup(content, 'html.parser')
	push_list = soup.find_all('div', class_= 'push')
	if push_list:
		for push in push_list:
			push_type = push.find('span')
			#determine if it's like or boo or None
			if push_type:
				push_type = push_type.text
				push_user = push.find('span', class_ = 'f3 hl push-userid').text
				p_list.append({'type':push_type, 'user':push_user})
		return p_list
	return ''
#==================for popular func.========================================
def parse_popular_info(content):
	
	p_list = []
	soup = BeautifulSoup(content, 'html.parser')
	main_content = soup.find(id='main-content')
	if main_content:
		main_content = main_content.find_all('a')
	else:
		main_content = 'None'
	for m in main_content:
		str_ = re.findall('([\'|\"]http.*(?:\.jpg|\.png|\.gif|\.jpeg)[\'|\"])', str(m))
		if str_ :
			p_list.append(str_[0][1:-1])
	if p_list:
		return p_list
	else:
		return ''
#==================for keyword func.========================================		
def parse_keyword_info(content):

	p_list = []
	keyword = sys.argv[2]
	soup = BeautifulSoup(content, 'html.parser')
	# get the text of main content
	contain = soup.findAll('div', class_ = 'push')
	if contain:
		for item in contain:
			item = item.extract()
	contain = soup.findAll('span', class_ = 'f2')		
	if contain:
		for item in contain:
			item = item.extract()
	content_tmp = soup.find(id='main-content')
	if content_tmp:
		main_content = content_tmp.text.lstrip().rstrip()
	else:
		main_content = 'None'
	#determine if the keyword is in the content
	if keyword in main_content:
		#get the whole URL
		soup2 = BeautifulSoup(content, 'html.parser')
		main_content = soup2.find(id='main-content')
		if main_content:
			main_content = main_content.find_all('a')
		else:
			main_content = 'None'		
		for m in main_content:
			str_ = re.findall('([\'|\"]http.*(?:\.jpg|\.png|\.gif|\.jpeg)[\'|\"])', str(m))
			if str_ :
				p_list.append(str_[0][1:-1])
	if p_list:
		return p_list
	else:
		return ''
#===============deal with files================================================
def parse_file(f, start_date, end_date):
	start_flag = False
	lines = f.readlines()
	_list = []
	for article in lines:
		ar_ = article.split(',')
		ar_info = {'date':convert_date(ar_[0]), 'URL':ar_[-1].strip('\n')}
		#start crawling from which artilce
		if start_flag == False and int(ar_info['date']) >= int(start_date):
			start_flag = True
		if start_flag == True:
			if int(ar_info['date']) > int(end_date): # end crawling
				break
			else:
				_list.append(ar_info['URL'].strip('\n'))
	return _list

def output_file(file_name, output_list):
	outfile = open(file_name, 'w', encoding = 'UTF-8')
	outfile.write("\n".join(output_list))

def convert_date(date):
	tmp = date.split('/')
	return int(tmp[0])*100+int(tmp[1])

def multiprocess(func, par):
	global process_num
	results =[]
	with Pool(processes=process_num) as pool:
		#一次request多次
		results = pool.map(func,  par)
	pool.close()
	pool.join()	
	return results	

#=====================================================================
#=========================function 1==================================
#=====================================================================
def crawl():
	all_page = []
	articles_info = []
	output_articles = []
	output_populars = []
	start = time.time()
	#1992  349
	for i in range(349):
		next_page_URL = "/bbs/Beauty/index" + str(i+1992)+".html"
		all_page.append(ptt_URL + next_page_URL)
	all_req = multiprocess(crawl_,  all_page)
	all_article_info = multiprocess(parse_article_info,  all_req)
	for ar_ in all_article_info:
		for aa in ar_:
			if aa['title'][0:4] != '[公告]':
				articles_info.append({'title': aa['title'], 'push' : aa['push'], 'date': aa['date'], 'link_URL':aa['link_URL']})
	outputs_info = multiprocess(parse_contents_info,  articles_info)

	while '' in outputs_info:
		outputs_info.remove('')
	for i in outputs_info:
		if i[0] == 'P':
			output_populars.append(i[1:])
		output_articles.append(i[1:])
	output_file('all_articles.txt', output_articles)
	output_file('all_popular.txt', output_populars)
	print('花費: %f 秒' % (time.time() - start))

#=====================================================================
#=========================function 2==================================
#=====================================================================
def get_rank(_list, _num):
	sort_list = []
	rank_list = []
	_set = set(_list)
	for item in _set:
		sort_list.append([-_list.count(item), item])
	sort_list.sort()
	for i in range(0, _num):
		rank_list.append( [sort_list[i][1], -sort_list[i][0]] )
	return rank_list


def push(start_date, end_date):

	parse_URL_list = []
	like_tmp_list = []
	boo_tmp_list = []
	rank_like = []
	rank_boo = []
	output_list = []

	rank_num = 10
	boo = 0
	like = 0
	start = time.time()
	f = open('all_articles.txt', 'r', encoding = 'utf8')
	parse_URL_list = parse_file(f, start_date, end_date)
	all_req = multiprocess(crawl_,  parse_URL_list)
	all_push_info = multiprocess(parse_push_info,  all_req)

	for i in all_push_info:
		for j in i:
			if '推' in j['type'] :
				like = like + 1
				like_tmp_list.append(j['user'])
			elif '噓' in j['type']:
				boo = boo + 1
				boo_tmp_list.append(j['user'])

	# get top 10 of like and boo
	if like_tmp_list:
		rank_like = get_rank(like_tmp_list, rank_num)
	if boo_tmp_list:
		rank_boo = get_rank(boo_tmp_list, rank_num)

	# construct output string list
	output_list.append('all like: '+str(like))
	output_list.append('all boo: '+str(boo))
	for i in range(0, rank_num):
		if (i+1) <= len(rank_like):
			output_list.append('like #' + str(i+1) + ': ' + rank_like[i][0] + ' ' + str(rank_like[i][1]))
		else:
			break
	for i in range(0, rank_num):
		if (i+1) <= len(rank_boo):
			output_list.append('boo #' + str(i+1) + ': ' + rank_boo[i][0] + ' ' + str(rank_boo[i][1]))
		else:
			break
	output_file('push['+str(start_date)+'-'+str(end_date)+'].txt', output_list)
	print('花費: %f 秒' % (time.time() - start))
#=====================================================================
#=========================function 3==================================
#=====================================================================
def popular(start_date, end_date):
	num_pop_article = 0;
	pop_pic_URL_list = [];
	output_list = []
	start = time.time()
	f = open('all_popular.txt', 'r', encoding = 'utf8')
	pop_pic_URL_list = parse_file(f, start_date, end_date)
	num_pop_article = len(pop_pic_URL_list)
	all_req = multiprocess(crawl_,  pop_pic_URL_list)
	all_pop_URL = multiprocess(parse_popular_info,  all_req)

	while '' in all_pop_URL:
		all_pop_URL.remove('')
	for i in all_pop_URL: 
		for j in i:
			output_list.append(j)

	# construct output string list
	output_list.insert(0, 'number of popular articles: ' + str(num_pop_article))
	output_file('popular['+str(start_date)+'-'+str(end_date)+'].txt', output_list)
	print('花費: %f 秒' % (time.time() - start))
#=====================================================================
#=========================function 4==================================
#=====================================================================
def keyword(keyword, start_date, end_date):
	search_pic_URL_list = [];
	output_list = []
	start = time.time()
	f = open('all_articles.txt', 'r', encoding = 'utf8')
	search_pic_URL_list = parse_file(f, start_date, end_date)
	all_req = multiprocess(crawl_,  search_pic_URL_list)
	all_keyword_URL = multiprocess(parse_keyword_info,  all_req)
	
	while '' in all_keyword_URL:
		all_keyword_URL.remove('')
	for i in all_keyword_URL: 
		for j in i:
			output_list.append(j)

	# construct output string list
	output_file('keyword('+keyword+')['+str(start_date)+'-'+str(end_date)+'].txt', output_list)
	print('花費: %f 秒' % (time.time() - start))
#=====================================================================
#=====================================================================
def check_date(start_date, end_date):
	st_m = int(start_date[0:-2])
	st_d = int(start_date[-2:])
	end_m = int(end_date[0:-2])
	end_d = int(end_date[-2:])
	m_ind1 = months.index(st_m)
	m_ind2 = months.index(end_m)
	if st_d > days[m_ind1] or end_d > days[m_ind2]:
		print("Input date error!!")
		exit()
	if st_m > end_m :
		print("Input date error!!")
		exit()
	elif st_m == end_m:
		if st_d	> end_m:
			exit()

if __name__ == "__main__":
	
	if sys.argv[1]:
		if sys.argv[1] == "crawl":
			crawl()
		elif sys.argv[1] == "push":
			check_date(sys.argv[2], sys.argv[3])
			push(sys.argv[2], sys.argv[3])
		elif sys.argv[1] == "popular":
			check_date(sys.argv[2], sys.argv[3])			
			popular(sys.argv[2], sys.argv[3])
		elif sys.argv[1] == "keyword":
			check_date(sys.argv[3], sys.argv[4])			
			keyword(sys.argv[2], sys.argv[3], sys.argv[4])
		else:
			print ("Error function Input!!")
			exit()