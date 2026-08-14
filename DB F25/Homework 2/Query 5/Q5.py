from setup import con, print_table

# Replace this with the instructor name you want to calculate for
instructor_name = "Dr. Smith"

query = f"""
WITH teaching_hours AS (
    SELECT  p.PersonID,
            COUNT(*) * 1.5 AS teach_hrs
    FROM    Person p
    JOIN    CodingClass c ON c.FacilitatorID = p.PersonID
    WHERE   p.FName || ' ' || p.LName = '{instructor_name}'
    GROUP BY p.PersonID
),
supervision_hours AS (
    SELECT  p.PersonID,
            COUNT(DISTINCT g.GroupID) * 4 * 1.5 AS supervise_hrs
    FROM    Person p
    JOIN    Project_Supervisor s ON s.PersonID = p.PersonID
    JOIN    Project_Group g ON g.GroupID = s.GroupID
    WHERE   p.FName || ' ' || p.LName = '{instructor_name}'
    GROUP BY p.PersonID
)
SELECT  p.PersonID,
        p.FName || ' ' || p.LName AS instructor,
        COALESCE(t.teach_hrs,0)      AS teaching_hours,
        COALESCE(s.supervise_hrs,0)  AS supervision_hours,
        COALESCE(t.teach_hrs,0)  * 35 AS teaching_pay,
        COALESCE(s.supervise_hrs,0) * 45 AS supervision_pay,
        (COALESCE(t.teach_hrs,0)  * 35 +
         COALESCE(s.supervise_hrs,0) * 45) AS total_pay
FROM        Person p
LEFT JOIN   teaching_hours    t ON t.PersonID = p.PersonID
LEFT JOIN   supervision_hours s ON s.PersonID = p.PersonID
WHERE       p.FName || ' ' || p.LName = '{instructor_name}';
"""

result = con.execute(query).fetchdf()
print(f"Q5. Given an instructor X, we want to know how much he/she got paid. Instructor: {instructor_name}")
print_table(result)
