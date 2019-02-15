import gcs_client

credentials_file = 'K9Bucket-2-b6152a9b63fe.json'
project_name = 'k9bucket-2'

credentials = gcs_client.Credentials(credentials_file)
project = gcs_client.Project(project_name, credentials)

buckets = project.list()[0]

print 'Contents of bucket %s:' % buckets
# if objects:
#     print '\t','\n\t'.join(map(lambda o: o.name + ' has %s bytes' % o.size, objects))
# else:
#     print '\tThere are no objects'