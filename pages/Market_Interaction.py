from streamlit_agraph import agraph, Node, Edge, Config
import streamlit as st


st.write('US Market only')
nodes = []
edges = []



nodes.append( Node(id="Market", 
                   size=25,
                   shape="start",
                   label="Market",
                   image=""
                   ) 
            )


nodes.append( Node(id="Utilities", 
                   label="Utilities", 
                   size=25, 
                   shape="start",
                   image="") 
            ) # includes **kwargs
nodes.append( Node(id="Real Estate", 
                   size=25,
                   shape="circularImage",
                   label="Real Estate",
                   image="",) 
            )

nodes.append( Node(id="Consumer Staples", 
                   size=25,
                   shape="circularImage",
                   label="Consumer Staples",
                   image=""
                   ) 
            )

nodes.append( Node(id="Consumer Discrentionary", 
                   size=25,
                   shape="circularImage",
                   label="Consumer Discrentionary",
                   image=""
                   ) 
            )
nodes.append( Node(id="Energy", 
                   size=25,
                   shape="circularImage",
                   label="Energy",
                   image=""
                   ) 
            )
nodes.append( Node(id="Communication Services", 
                   size=25,
                   shape="circularImage",
                   label="Communication Services",
                   image=""
                   ) 
            )

nodes.append( Node(id="Industrials", 
                   size=25,
                   shape="circularImage",
                   label="Industrials",
                   image=""
                   ) 
            )

nodes.append( Node(id="Finance", 
                   size=25,
                   shape="circularImage",
                   label="Finance",
                   image=""
                   ) 
            )


nodes.append( Node(id="Basic Materials", 
                   size=25,
                   shape="circularImage",
                   label="Basic Materials",
                   image=""
                   ) 
            )

nodes.append( Node(id="Technology", 
                   size=25,
                   shape="circularImage",
                   label="Technology",
                   image=""
                   ) 
            )


######################
#####################
#edges
####################
######################
edges.append( Edge(source="Real Estate", 
                   label="", 
                   target="Utilities", 
                   color='green',smooth=False,
                   arrows='False'
                   # **kwargs
                   ) 
            ) 

edges.append( Edge(source="Consumer Staples", 
                   label="", 
                   target="Utilities", 
                   color='green',smooth=False,
                   arrows='False'
                   # **kwargs
                   ) 
            ) 

################
################

edges.append( Edge(source="Energy", 
                   label="", 
                   target="Market", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            ) 
edges.append( Edge(source="Market", 
                   label="", 
                   target="Energy", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            ) 


edges.append( Edge(source="Finance", 
                   label="", 
                   target="Market", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            ) 
edges.append( Edge(source="Consumer Discrentionary", 
                   label="", 
                   target="Market", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            ) 


edges.append( Edge(source="Technology", 
                   label="", 
                   target="Health Care", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )
edges.append( Edge(source="Technology", 
                   label="", 
                   target="Market", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )
edges.append( Edge(source="Market", 
                   label="", 
                   target="Technology", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )




edges.append( Edge(source="Industrials", 
                   label="", 
                   target="Technology", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )

 

edges.append( Edge(source="Industrials", 
                   label="", 
                   target="Consumer Discretionary",
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )  



edges.append( Edge(source="Finance", 
                   label="", 
                   target="Industrials", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )  


edges.append( Edge(source="Basic Materials", 
                   label="", 
                   target="Industrials", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            ) 



edges.append( Edge(source="Market", 
                   label="", 
                   target="Communication Services", 
                   color='green',smooth=True,type='CurvedCW'
                   # **kwargs
                   ) 
            )

config = Config(width=700,
                height=700,
                directed=True, 
                physics=True, 
                hierarchical=True,

                # **kwargs
                )

return_value = agraph(nodes=nodes, 
                      edges=edges, 
                      config=config)
