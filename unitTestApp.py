import unittest
from flask import json, session
from chat import app, allowed_file

class FlaskChatbotTestCase(unittest.TestCase):

    def setUp(self):
        # Set up a test client
        self.app = app
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False
        self.client = self.app.test_client()

    def login(self, username, password):
        return self.client.post('/login', data=dict(
            username=username,
            password=password
        ), follow_redirects=True)

    def logout(self):
        print( self.client.get('/logout', follow_redirects=True))

    def test_index_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Chatbot', response.data)
        print("Test index page passed.")

    def test_get_response(self):
        response = self.client.post('/get_response', data=json.dumps({
            'question': 'How to get admission?'
        }), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('response', data)
        print("Test get response passed.")

    def test_login_logout(self):
        # Test login
        response = self.login('admin', '123')
        self.assertIn(b'Admin Page', response.data)
        self.assertTrue('username' in session)
        print("Test login passed.")

        # Test logout
        response = self.logout()
        self.assertNotIn('username', session)
        print("Test logout passed.")

    def test_login_required_redirect(self):
        response = self.client.get('/admin', follow_redirects=True)
        self.assertIn(b'Please log in to access this page.', response.data)
        print("Test login required redirect passed.")

    def test_allowed_file(self):
        self.assertTrue(allowed_file('test.pdf'))
        self.assertFalse(allowed_file('test.txt'))
        print("Test allowed file passed.")

    def test_add_intent(self):
        self.login('admin', '123')
        response = self.client.post('/add_intent', data=dict(
            tag='test_tag',
            patterns='Hello\nHi',
            responses='Hello there\nHi there',
            context_set=''
        ), follow_redirects=True)
        # self.assertIn(b'Admin Page', response.data)
        print("Test add intent passed.")

    def test_delete_intent(self):
        self.login('admin', '123')
        response = self.client.post('/delete_intents', data=dict(
            intents_to_delete=['test_tag']
        ), follow_redirects=True)
        # self.assertIn(b'Admin Page', response.data)
        print("Test delete intent passed.")

    def test_upload_file(self):
        self.login('admin', '123')
        with open('test.pdf', 'rb') as pdf:
            response = self.client.post('/upload', content_type='multipart/form-data', data=dict(
                file=(pdf, 'test.pdf')
            ), follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            print("Test upload file passed.")

    def test_update_response(self):
        self.login('admin', '123')
        response = self.client.post('/update_response', data=dict(
            response_greeting='Hello there!'
        ), follow_redirects=True)
        self.assertIn(b'Admin Page', response.data)
        print("Test update response passed.")

    def test_list_files(self):
        self.login('admin', '123')
        response = self.client.get('/list_files')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('files', data)
        print("Test list files passed.")

    def test_delete_files(self):
        self.login('admin', '123')
        response = self.client.post('/delete_files', data=json.dumps({
            'file_urls': ['http://127.0.0.1:5000/static/uploads/test.pdf']
        }), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        print("Test delete files passed.")

    def tearDown(self):
        print("Tearing down the test case environment.")
        pass

test = FlaskChatbotTestCase()
test.setUp()
test.test_index_page()
test.login('admin','123')
test.test_upload_file()
test.test_add_intent()
test.test_delete_files()
test.test_delete_intent()
test.test_list_files()



