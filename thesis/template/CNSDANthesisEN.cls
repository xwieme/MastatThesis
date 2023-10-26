% CLASS file for wethesis
% Based on the book class from Latex
% author: Joris Meys
% 2017/05/29

% --- Class structure identification

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{CNSDANthesisEN}[2018/03/09 v1.1 CNSDANthesisEN]
% --- Class structure declaration of options
%
%  This class extends book class

% Color definitions for titles. The actual colors are defined
% in bwtitlepage
\newcommand{\coltitle}{\color{black}}
\newcommand{\coltitlepage}{\color{UGBlue}}

% Add new option
\DeclareOption{usecolor}{
	\renewcommand{\coltitle}{\color{UGBlue}}
}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}

% --- Class structure execution of options
%
\ProcessOptions \relax

\LoadClass{book}

% --- Loading necessary packages
\RequirePackage[margin = 2.5cm]{geometry}
\RequirePackage{CNSDANtitlepageEN} % To construct the title page
\RequirePackage{caption} % To adjust captions

% --- adjust binding offset
\if@twoside\geometry{bindingoffset = 1cm}\fi

% --- adjust captions
\captionsetup[table]{skip=1.5ex}

% --- Define the standard lettertype: Arev

\RequirePackage{arev}

% --- Set up the layout parameters

% set interline set to ...
\renewcommand{\baselinestretch}{1.5}
% no indentations
\setlength{\parindent}{0cm}
% more space between paragraphs
\setlength{\parskip}{2.3ex plus 0.3ex minus 0.3ex}
% makes all pages the height of the text on that page, and no extra vertical space is added
\raggedbottom

% define tocdepth
\setcounter{tocdepth}{2}
% set lenght of headheight
\setlength{\headheight}{21pt}
% to suppress widows and orphans
\widowpenalty=10000
\clubpenalty=10000

% --- Set up the headers and footers
\RequirePackage{fancyhdr}

% for the frontmatter
\fancypagestyle{frontmatter}{
	\fancyhf{}
	\renewcommand{\headrulewidth}{0pt}
	\fancyfoot[C]{\thepage}
}
% for the mainmatter
\fancypagestyle{mainmatter}{
	\fancyhf{}
	\fancyhead[LO]{ \leftmark}
	\fancyhead[RE]{ \rightmark}
	\fancyfoot[C]{\thepage}
	\renewcommand{\headrulewidth}{0.15pt}
}
% for the title pages
\fancypagestyle{plain}{
	\fancyhf{}
	\renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}


% --- Set up the front matter
\renewcommand{\listfigurename}{List of Figures}
\renewcommand{\listtablename}{List of Tables}

% --- Set up Chapter titles etc
\RequirePackage[raggedright, explicit]{titlesec}

\titleformat{\chapter} % command
  [display] % shape
  {\fontsize{18}{22} \selectfont \coltitle } % format
  {\MakeUppercase{\chaptertitlename \ \thechapter}} % the label
  {-2ex} %separator space
  {\fontsize{24}{32} \selectfont \bf \raggedright \MakeUppercase{\uline{#1}}} %before code
  { } %after code

% Change the spacing above the chapter. THis is left - above - below
\titlespacing*{\chapter}{0pt}{0pt}{40pt}

% --- Setup the document

\AtBeginDocument{

\maketitle

}
