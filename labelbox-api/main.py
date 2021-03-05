from flask import Flask, redirect, url_for, request, jsonify, render_template
from labelbox import Client, Project, Dataset
from typing import Dict
from datetime import datetime
from collections import defaultdict
from bson.objectid import ObjectId
from os.path import join, dirname
from dotenv import load_dotenv

import ndjson
import requests

import json
import os
import pymongo 
import uuid
import re

app = Flask(__name__)

labelbox_api_key = os.environ.get('LABELBOX_API_KEY')
mongo_api_key = os.environ.get('MONGO_DB')
token_checker = os.environ.get('TOKEN_CHECKER')

# Connect client
client = Client(api_key=str(labelbox_api_key))

mongo_client = pymongo.MongoClient(str(mongo_api_key))
mydb = mongo_client["omdena"]
mycol = mydb["upload_annotations"]

def get_project_uid(project_name):
  projects = client.get_projects(where=(Project.name == project_name))
  project = next(iter(projects))
  project_uid = project.uid
  return project_uid

def get_ontology(client: Client, project_id: str) -> Dict[str, str]:
  result = client.execute("""
    query get_ontology($proj_id: ID!) {
      project(where: {id: $proj_id}) {
        ontology {
          normalized
          }
          }
      }
    """, 
    {"proj_id": project_id})
  
  return result['project']['ontology']['normalized']


def get_entity_schema_id(ontology, name):
  tools = ontology['tools']
  return next((t for t in tools if t["name"] == name), None)

def get_current_import_requests(project_id):
    response = client.execute(
                    """
                    query get_all_import_requests(
                        $project_id : ID! 
                    ) {
                      bulkImportRequests(where: {projectId: $project_id}) {
                        id
                        name
                        state
                      }
                    }
                    """,
                    {"project_id": project_id})
    
    return response['bulkImportRequests']

def delete_import_request(import_request_id):
    response = client.execute(
                    """
                        mutation delete_import_request(
                            $import_request_id : ID! 
                        ){
                          deleteBulkImportRequest(where: {id: $import_request_id}) {
                            id
                            name
                          }
                        }
                    """,
                    {"import_request_id": import_request_id})
    
    return response

@app.route('/upload-annotation-list', methods = ['POST', 'GET'])
def upload_annotation_list():
    result = {}
    try:
      result['data'] = []

      for col in mycol.find():
        task_id = "---"
        if 'task_id' in col:
          task_id = col['task_id']

        result['data'].append({
          'id': str(col['_id']),
          'project': col['project'],
          'date': col['date'],
          'upload_name': col['upload_name'],
          'submitted_by': col['submitted_by'],
          'description': col['description'],
          'status': col['status'],
          'task_id': task_id,
          'data': col['data'],
        })
      
      result['status'] = 'Successfully Uploaded Annotations'
    except Exception as e:
      result['status'] = 'Error Uploading'
      result['message'] = str(e)

    return json.dumps(result)


@app.route('/delete', methods = ['GET'])
def delete():
    result = {}
    try:
      id = request.args.get('id')
      current_data = mycol.find_one({"_id": ObjectId(id)})

      delete_import_request(current_data['task_id'])
      
      myquery = {"_id": ObjectId(id)}
      newvalues = { "$set": { "status": 'DELETED'}}
      mycol.update_one(myquery, newvalues)

      result['status'] = 'Successfully Deleted'
    except Exception as e:
      result['status'] = 'Error Deleting'
      result['message'] = str(e)


    return json.dumps(result)



@app.route('/update-status', methods = ['GET'])
def update_status():
    result = {}
    try:
      id = request.args.get('id')

      current_data = mycol.find_one({"_id": ObjectId(id)})
      if len(current_data) > 0:
        if current_data['status'] == 'RUNNING':
          upload_name = current_data['upload_name']

          # Get correct project
          project_name = current_data['project']
          project_uid = get_project_uid(project_name)
          all_import_requests = get_current_import_requests(project_uid)

          task_id = 0
          status = "RUNNING"
          for _all_import_requests in all_import_requests:
            if upload_name == _all_import_requests['name'].strip():
              task_id = _all_import_requests['id']
              status = _all_import_requests['state']
              break

        result['task_id'] = task_id
        result['status'] = status

        if task_id  != 0:
          myquery = {"_id": ObjectId(id)}
          newvalues = { "$set": { "status": status, "task_id": task_id}}
          mycol.update_one(myquery, newvalues)

    except Exception as e:
      result['status'] = 'Error Deleting'
      result['message'] = str(e)

    return json.dumps(result)


@app.route('/submit-annotation', methods = ['POST', 'GET'])
def submit_annotation():
    
    """
    user : Username who is submitting annotation
    description : short description
    project : project name to annotate in labelbox
    token : secret token set in heroku env variable
    data : list of annnotations

    POST Request To Submit annotation
    {
      'user' : str,
      'description' : str,
      'project' : str,
      'token' : str
      'data' : Annotations(list)
    }

    Single Annotation format -
    {
      'tag' : feature_name(str),
      'pmid' : pmid(str),
      'location' : {
        'start' : integer,
        'end' : integer
      }
    }
    """
    received_data = json.loads(request.data)
    annotations = []

    if request.method == 'POST':
        if received_data['token'] == token_checker:
          mydb = mongo_client["omdena"]
          mycol = mydb["upload_annotations"]

          # Get project details
          project_name = received_data['project']
          projects = client.get_projects(where=(Project.name == project_name))
          project = next(iter(projects))
          project_name = project.name
          project_uid = project.uid

          # Get associated dataset
          dataset = next(iter(project.datasets()))
          dataset_name = dataset.name
          dataset_uid = dataset.uid

          # Get ontology for the project
          ontology = get_ontology(client, project.uid)

          rows = list(dataset.data_rows())
          mapping = defaultdict()

          # Create mapping pmid->row uid
          for row in rows:
            abstract = row.row_data
            pmid_begin = re.search(r"\d", abstract).start()
            pmid_end = abstract.find('T') - 1
            pmid = str(abstract[pmid_begin:pmid_end])

            mapping[pmid] = row.uid

        
          # Iterate through submitted annotations
          for _received_data in received_data['data']:
            schema_id = get_entity_schema_id(ontology, _received_data['tag'])
            annotations.append({
              'uuid':str(uuid.uuid4()), 
              "schemaId":schema_id['featureSchemaId'], 
              "dataRow": { 
                  "id": mapping[_received_data['pmid']]
              }, 
              "location": { 
                  "start": _received_data['location']['start'], 
                  "end": _received_data['location']['end'],  
              }
            })
          
          upload_name = '{}_{}_{}'.format(project_name, received_data['user'], datetime.now().strftime("%d/%m/%Y %H:%M:%S").replace(" ", "_"))
          project.upload_annotations(
            name=upload_name, 
          annotations=annotations)

          mycol.insert_one({
            'project': received_data['project'],
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'submitted_by': received_data['user'], 
            'description': received_data['description'],
            'upload_name': upload_name,
            'status': 'RUNNING',
            'data': annotations
          })

          return json.dumps(annotations)
        else:
          return json.dumps({'message':'Invalid Token'})
        

    


@app.route('/available-abstracts', methods = ['POST', 'GET'])
def available_abstracts():
    result = {}
    mydata = mydb["abstract_details"]
    if request.method == 'GET':
      result['data'] = []
      for _mydata in mydata.find():
        result['data'].append({
          'row_data': _mydata['row_data'],
          'uid': _mydata['uid'],
          'external_id': _mydata['external_id']
        })
    else:
      received_data = json.loads(request.data)
      current_data = mydata.find_one({"row_data": received_data['row_data'].strip()})
      if current_data is not None:
        result['row_uid'] = current_data['uid']
        result['external_id'] = current_data['external_id']


    return json.dumps(result)


@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')


@app.route('/search', methods = ['GET'])
def search():
    return render_template('search.html')


if __name__ == '__main__':
   app.run(debug = True)