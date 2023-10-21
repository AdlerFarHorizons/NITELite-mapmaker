# Tips on Data Handling from the Zooniverse Team
(Zooniverse Zach and Cliff)

A single file that the user deals with is much easier than multiple files.

AWS is more than capable of handling all the processing in the cloud.
However, Zooniverse currently in Azureshop, so they don't have as many
tips about AWS.
That said, Lambda (in AWS) is good for running serverless functions.
It can run docker containers.
It can have restful interfaces.
Lambda triggers when you upload, and then shuts down afterwards,
which is nice.
However, there are many ways to run *a* container on AWS.
Zooniverse used to run all of their services in ec2,
which ran virtual machines,
which ran docker,
which ran app containers.
Great reference: https://github.com/zooniverse/theia/

For making this pipeline last, the first step is containerization,
e.g. building a docker image s.t. compute environments are not a concern.
Then the question becomes, where does this container live and how is it run?
Could draw up an amazon machine image (AMI) that sets up everything and runs it.

Regarding data access, whatever it is, just mount S3 as a data volume.

Docker is layers.
Base layer is Python at a specific version.
Then, add on your own layers.
Apt-get-install packages, etc.
