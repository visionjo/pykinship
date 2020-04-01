## DB Contents and Structure
- FIW_PIDs.csv:
  - Photo lookup table. Each row is an image instance, containing the following fields:
     - PID: Photo ID
     - Name: Surname.firstName (root reference for given family)
     - URL: Photo URL on web
     - Metadata: Text caption for photo
- FIW_FIDs.csv:FID (family)/ Surname lookup table.
  - FID:Unique ID key assigned to each family.
  - Surname:Family Name corresponding to FID key.
- FIW_RIDs.csv:
  - Relationship lookup table with keys \[1-9\] assigned to relationship types.
- FIDs/
  - FID####/ Contains labels and cropped facial images for members of family (1-1000)
    - MID#/ Face images of family member with ID key <N>, i.e., MID #.
    -  mid.csv:
       - File containing member information of each family:
          - relationship matrix
          - first name of family member.
          - gender (m / f)


For example:
```
FID0001.csv
    
	0     1     2     3     Name    Gender
	1     0     4     5     name1   female
	2     1     0     1     name2   female
	3     5     4     0     name3   male
	
```
Here we have 3 family members, as seen by the MIDs across columns and down rows.


We can see that MID1 is related to MID2 by 4->1 (Parent->Sibling), which of course can be viewed as the inverse, i.e., MID2->MID1 is 1->4. It can also be seen that MID1 and MID3 are Spouses of one another, i.e., 5->5. And so on, and so forth.
