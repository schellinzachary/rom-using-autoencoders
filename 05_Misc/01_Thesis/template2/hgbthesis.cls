%% hgbthesis.cls
%% LaTeX template for creating diploma theses, masters's theses, bachelor's theses and other term papers.
%% This file is part of the Hagenberg Thesis package for creating bachelors's and masters's theses
%% Author: Wilhelm Burger (wilhelm.burger@fh-hagenberg.at)
%% GitHub: https://github.com/Digital-Media/HagenbergThesis
%% File encoding: ASCII

%% Package Information --------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hgbthesis}[2017/10/28]

\RequirePackage{xifthen}

\RequirePackage{ifpdf}	% for checking PDF mode
\ifpdf{}% \typeout{hgbthesis: *** LaTeX in PDF mode ***}
\else{\errmessage{hgbthesis: *** LaTeX NOT running in PDF mode ***}}
\fi


%% Commands for Title Pages (used in class options) ---------------------------

\newcommand{\hgb@MainLanguage}{german}	% also referenced in hgb.sty
\newcommand{\@mastertitlepages}{}
\newcommand{\@diplomatitlepages}{}
\newcommand{\@bachelortitlepages}{}
\newcommand{\@internshiptitlepages}{}
\newcommand{\@maketitlepages}{\@mastertitlepages}  %Default: Master's thesis
\newcommand{\HRule}{\rule{\linewidth}{0.5mm}} % Defines a new command for horizontal lines, change thickness here


\newcommand{\@titlepagefont}{\sffamily}

%% Class options --------------------------------------------------------------

\DeclareOption{master}{
	\renewcommand{\@maketitlepages}{\@mastertitlepages}
}

\DeclareOption{diplom}{%
	\renewcommand{\@maketitlepages}{\@diplomatitlepages}
	\ClassWarning{hgbthesis}{Deprecated: Document option '\CurrentOption' is deprecated, use 'diploma' instead.}
}

\DeclareOption{diploma}{%
	\renewcommand{\@maketitlepages}{\@diplomatitlepages}
}

\DeclareOption{bachelor}{%
	\renewcommand{\@maketitlepages}{\@bachelortitlepages}
}

\DeclareOption{praktikum}{%
	\renewcommand{\@maketitlepages}{\@internshiptitlepages}
	\ClassWarning{hgbthesis}{Deprecated: Document option '\CurrentOption' is deprecated, use 'internship' instead.}
}

\DeclareOption{internship}{%
	\renewcommand{\@maketitlepages}{\@internshiptitlepages}
}

\DeclareOption{english}{%
	\renewcommand{\hgb@MainLanguage}{english}
}

\DeclareOption{german}{%
	\renewcommand{\hgb@MainLanguage}{german}
}

\DeclareOption*{%
	\ClassWarning{hgbthesis}{Unknown document option '\CurrentOption'.}
}

\ProcessOptions


%% Paper Options and Style File -----------------------------------------------

\LoadClass[a4paper,11pt,twoside]{scrbook}
\RequirePackage[top=36mm,bottom=40mm,left=32mm,right=32mm]{geometry} %showframe

\RequirePackage{latex/hgb}	% load associated style file hgb.sty


%% Setup for Pages in the Front Matter ----------------------------------------

% General Commands

\newcommand{\@programname}{\@MissingArg{Degree program not specified!}}
\newcommand{\studiengang}[1]{%
	\renewcommand{\@programname}{#1}
	\ClassWarning{hgbthesis}{Deprecated: '\string\studiengang' is deprecated, use '\string\programname' instead.}
}
\newcommand{\programname}[1]{%
	\ifthenelse{\isempty{#1}}{}{\renewcommand{\@programname}{#1}}
}

\newcommand{\@placeofstudy}{\@MissingArg{Place of study not specified!}}
\newcommand{\studienort}[1]{%
	\renewcommand{\@placeofstudy}{#1}
	\ClassWarning{hgbthesis}{Deprecated: '\string\studienort' is deprecated, use '\string\placeofstudy' instead.}
}
\newcommand{\placeofstudy}[1]{%
	\ifthenelse{\isempty{#1}}{}{\renewcommand{\@placeofstudy}{#1}}
}

\newcommand{\@advisor}{\@MissingArg{Advisor not specified!}}
\newcommand{\betreuer}[1]{%
	\renewcommand{\@advisor}{#1}
	\ClassWarning{hgbthesis}{Deprecated: '\string\betreuer' is deprecated, use '\string\advisor' instead.}
}
\newcommand{\betreuerin}[1]{%
	\renewcommand{\@advisor}{#1}
	\ClassWarning{hgbthesis}{Deprecated: '\string\betreuerin' is deprecated, use '\string\advisor' instead.}
}
\newcommand{\advisor}[1]{%
	\ifthenelse{\isempty{#1}}{}{\renewcommand{\@advisor}{#1}}
}

\newcommand{\@thesisnumber}{\@MissingArg{Thesis number not specified!}}
\newcommand{\nummer}[1]{%
	\renewcommand{\@thesisnumber}{#1}
	\ClassWarning{hgbthesis}{Deprecated: '\string\nummer' is deprecated, use '\string\thesisnumber' instead.}
}
\newcommand{\thesisnumber}[1]{%
	\ifthenelse{\isempty{#1}}{}{\renewcommand{\@thesisnumber}{#1}}
}

\newcommand{\@coursetitle}{\@MissingArg{Course not specified!}}
\newcommand{\gegenstand}[1]{%
	\renewcommand{\@coursetitle}{#1}
	\ClassWarning{hgbthesis}{Deprecated: '\string\gegenstand' is deprecated, use '\string\coursetitle' instead.}
}
\newcommand{\coursetitle}[1]{%
	\ifthenelse{\isempty{#1}}{}{\renewcommand{\@coursetitle}{#1}}
}

\newcommand{\@semester}{\@MissingArg{Semester not specified!}}
\newcommand{\semester}[1]{%
	\ifthenelse{\isempty{#1}}{}{\renewcommand{\@semester}{#1}}
}

\newcommand{\@logofile}[0]{}
\newcommand{\logofile}[1]{%
	\renewcommand{\@logofile}{#1}
}

\newcommand{\makelogo}[0]{%
\ifthenelse{\equal{\@logofile}{}}%
	{\vspace*{30mm}}%
	{\includegraphics[height=30mm]{\@logofile}}%
}

% Submission Commands

\newcounter{@dayofsubmission}
\newcounter{@monthofsubmission}
\newcounter{@yearofsubmission}

\setcounter{@dayofsubmission}{\the\day}
\setcounter{@monthofsubmission}{\the\month}
\setcounter{@yearofsubmission}{\the\year}

\newcommand{\abgabedatum}[3]{%	%\abgabedatum{yyyy}{mm}{dd}
	\setcounter{@yearofsubmission}{#1}
	\setcounter{@monthofsubmission}{#2}
	\setcounter{@dayofsubmission}{#3}
	\ClassWarning{hgbthesis}{Deprecated: '\string\abgabedatum' is deprecated, use '\string\dateofsubmission{yyyy}{mm}{dd}' instead.}
}
\newcommand{\dateofsubmission}[3]{%	%\dateofsubmission{yyyy}{mm}{dd}
	\setcounter{@yearofsubmission}{#1}
	\setcounter{@monthofsubmission}{#2}
	\setcounter{@dayofsubmission}{#3}
}

\newcommand{\abgabemonat}[1]{%
	\ClassWarning{hgbthesis}{Deprecated: '\string\abgabemonat' is deprecated, use '\string\dateofsubmission{yyyy}{mm}{dd}' instead.}
}

\newcommand{\abgabejahr}[1]{%
	\ClassWarning{hgbthesis}{Deprecated: '\string\abgabejahr' is deprecated, use '\string\dateofsubmission{yyyy}{mm}{dd}' instead.}
}

% Internship Report Commands

\newcommand{\@companyName}{\@MissingArg{Company not specified!}}
\newcommand{\firma}[1]{%
	\renewcommand{\@companyName}{#1}
	\ClassWarning{hgbthesis}{Deprecated: '\string\firma' is deprecated, use '\string\companyName' instead.}
}
\newcommand{\companyName}[1]{%
	\ifthenelse{\isempty{#1}}{}{\renewcommand{\@companyName}{#1}}
}

\newcommand{\@companyUrl}{\@MissingArg{Company URL not specified!}}
\newcommand{\firmenUrl}[1]{%
	\renewcommand{\@companyUrl}{#1}
	\ClassWarning{hgbthesis}{Deprecated: '\string\firmenUrl' is deprecated, use '\string\companyUrl' instead.}
}
\newcommand{\companyUrl}[1]{%
	\ifthenelse{\isempty{#1}}{}{\renewcommand{\@companyUrl}{#1}}
}

\newcommand{\@companyPhone}{\@MissingArg{Company phone number not specified!}}
\newcommand{\firmenTel}[1]{%
	\renewcommand{\@companyPhone}{#1}
	\ClassWarning{hgbthesis}{Deprecated: '\string\firmenTel' is deprecated, use '\string\companyPhone' instead.}
}
\newcommand{\companyPhone}[1]{%
	\ifthenelse{\isempty{#1}}{}{\renewcommand{\@companyPhone}{#1}}
	\renewcommand{\@companyPhone}{#1}
}

\newcommand{\@license}{\@MissingArg{License information is missing!}}
\newcommand{\license}[1]{%
	\renewcommand{\@license}{#1}
}


%% Commands for Creating the Title Pages --------------------------------------

\newcommand{\titelseite}{%
	\maketitle
	\ClassWarning{hgbthesis}{Deprecated: '\string\titelseite' is deprecated, use '\string\maketitle' instead.}
}

%% adds blank space after each letter (http://stackoverflow.com/a/3951837)
\newcommand{\@addspaces}[1]{%
  \@tfor\letter:=#1\do{%
    \letter\thinspace%	%\space
  }%
}

\newcommand{\@mastertitle}{%
\begin{titlepage}%
	\begin{german}
		\rule{0mm}{1mm}
		\vspace*{0mm}
			%\@titlepagefont
			\begin{minipage}[h]{0.3\textwidth}
			\makelogo
			\end{minipage}%
			\begin{minipage}[h]{0.7\textwidth}
			\textsc{\LARGE Technische Universit\"at Berlin}\\[0.3cm] % Main heading such as the name of your
			{\Large  Institut f\"ur Str\"omungsmechnaik \\und Technische Akustik}\\[0.2cm] % Major heading such as course name
			{\Large Fachgebiet Numerische Fluiddynamik}\\
			\end{minipage}\\[2cm]%


			\begin{center}%
			% Submitted in fulfilment of the requirements for the degree of
			{\LARGE{Masterthesis}}%
			\normalfont
			\setlength{\unitlength}{1cm}
			\parbox[c][5cm][c]{14cm}{%
				\centering\LARGE\bfseries%%
				\setlength{\baselineskip}{25pt}%
				\HRule\\[0.4cm]
				\@title\\%
				\HRule\\
			}
			{\Large{submitted in fulfilment of the requirements\\ for the degree of}\\[4mm]
				\Large{\textit{M.~Sc. Physikalische Ingenieurwissenschaft}
			}}%
			\end{center}%

			\vskip 30mm%
			{\large\textbf{Author: }\\ \@author, 327752 }%
				\vskip 4mm%
			{\large\textbf{Submission: }\\Berlin, 23th July 2018}
			\vskip 4mm%
			{\large\textbf{Examiner:} \\%
			Prof.~Dr.~sc.~techn.~habil.~J\"orn Sesterhenn\\
			Prof.~Dr.~rer.~nat.~Julius Reiss
			}
			\vskip 16mm%%
		\vfil%
	\end{german}
\end{titlepage}%
}

% Copyright and License Page --------------------------------------------------

\newcommand{\strictlicense}{%
\ifthenelse{\equal{\hgb@MainLanguage}{german}}%
	{\license{Alle Rechte vorbehalten}}%
	{\license{All Rights Reserved}}
}

\newcommand{\@ccLicenseAt}{%
Diese Arbeit wird unter den Bedingungen der
\textsf{Creative Commons Lizenz}
\emph{Attribution-NonCommercial-NoDerivatives 4.0 International}
(CC BY-NC-ND 4.0) ver\"offentlicht --
siehe \url{https://creativecommons.org/licenses/by-nc-nd/4.0/}.}

\newcommand{\@ccLicenseEn}{%
This work is published under the conditions of the
\textsf{Creative Commons License}
\emph{Attribution-NonCommercial-NoDerivatives 4.0 International}
(CC BY-NC-ND 4.0)---see
\url{https://creativecommons.org/licenses/by-nc-nd/4.0/}.}

\newcommand{\cclicense}{%
\ifthenelse{\equal{\hgb@MainLanguage}{german}}%
	{\license{\@ccLicenseAt}} % german CC license text
	{\license{\@ccLicenseEn}} % english CC license text (default)
}

\cclicense	% Creative Commons License (default)

\newcommand{\@makelicense}{
	\sbox\@tempboxa{\@license}%
	\ifdim \wd\@tempboxa >\hsize
		\begin{sloppypar}\noindent\@license\end{sloppypar}
	\else
		\global \@minipagefalse
		\hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
	\fi
}

\newcommand{\@copyrightpage}{%
\newpage
\setcounter{page}{2}
\thispagestyle{plain}
\rule{0mm}{1mm}
\vspace*{75mm}
\begin{center}
\textsf{\copyright\ Copyright\ \arabic{@yearofsubmission}\ \@author}
\end{center}
\vspace{8mm}
\@makelicense
}


%% Company Page for Internship Report -----------------------------------------

\newcommand{\@companypage}{%
\newpage
\setcounter{page}{2}
\thispagestyle{plain}
\rule{0mm}{1mm}
\begin{center}
\vspace{25mm}
{Praktikumsstelle:}
\\
\vspace{3mm}
{\Large\sf\@companyName\\ \mbox{}}
\\
{\sf \@companyPhone}
\\
{\sf \@companyUrl}
\\
\vspace{45mm}
{Kontaktperson:}
\\
\vspace{3mm}
{\Large\sf\@advisor\\ \mbox{}}
\end{center}
}


%% Advisor Pages in German and English ----------------------------------------

% German

\newcommand{\@advisorpageDe}{%
	\newpage
	\setcounter{page}{2}
	\thispagestyle{plain}
	\rule{0mm}{1mm}
	\begin{center}
		%
		\vspace{40mm}
		{Diese Arbeit entstand im Rahmen des Gegenstands}\\
		\vspace{6mm}
		{\Large\sf \@coursetitle\\ \mbox{}}\\
		\medskip
		im\\
		\bigskip
		{\@semester}\\
		%
		\vspace{45mm}
		{Betreuung:}\\
		\vspace{3mm}
		{\Large\sf \@advisor\\ \mbox{}}
	\end{center}
}

% English

\newcommand{\@advisorpageEn}{%
	\newpage
	\setcounter{page}{2}
	\thispagestyle{plain}
	\rule{0mm}{1mm}
	\begin{center}
		%
		\vspace{40mm}
		{This thesis was created as part of the course}\\
		\vspace{6mm}
		{\Large\sf \@coursetitle\\ \mbox{}}\\
		\medskip
		during\\
		\bigskip
		{\@semester}\\
		%
		\vspace{45mm}
		{Advisor:}\\
		\vspace{3mm}
		{\Large\sf \@advisor\\ \mbox{}}
	\end{center}
}

\newcommand{\@advisorpage}{%
\ifthenelse{\equal{\hgb@MainLanguage}{german}}%
	{\@advisorpageDe}%
	{\@advisorpageEn}
}


%% Declaration Pages in German and English ------------------------------------

% German

\newcommand{\@declarationpageDe}{%
	\begin{german}
		\chapter{Erkl\"arung}
		\noindent
		%Standardfassung der FH-OOe ab 04.04.2012:
		Ich erkl\"are eidesstattlich, dass ich die vorliegende Arbeit selbstst\"andig und ohne fremde Hilfe verfasst,
		andere als die angegebenen Quellen nicht benutzt und die den benutzten Quellen entnommenen Stellen als
		solche gekennzeichnet habe. Die Arbeit wurde bisher in gleicher oder \"ahnlicher Form keiner anderen
		Pr\"ufungsbeh\"orde vorgelegt.
		\par
		\vspace{10mm}
		\noindent
		\@placeofstudy, am \formatdate{\value{@dayofsubmission}}{\value{@monthofsubmission}}{\value{@yearofsubmission}}
		\par
		\vspace{12mm}
		\noindent
		\@author
	\end{german}
}

% English

\newcommand{\leadingzero}[1]{\ifnum #1<10 0\the#1\else\the#1\fi}

\newcommand{\@declarationpageEn}{%
	\begin{english}
		\chapter*{Declaration}
		\noindent
		I hereby declare and confirm that this thesis is entirely the result of my own original work.
		Where other sources of information have been used, they have been indicated as such and properly
		acknowledged. I further declare that this or similar work has not been submitted for credit elsewhere.
		\par
		\vspace{10mm}
		\noindent
		%\@placeofstudy, \the\year-\leadingzero{\month}-\leadingzero{\day} %http://de.wikibooks.org/wiki/LaTeX-W%F6rterbuch:_today
		\@placeofstudy, {\usdate\formatdate{\value{@dayofsubmission}}{\value{@monthofsubmission}}{\value{@yearofsubmission}}}
		\par
		\vspace{12mm}
		\noindent
		\@author
	\end{english}
}

\newcommand{\@declarationpage}{%
\ifthenelse{\equal{\hgb@MainLanguage}{german}}%
	{\@declarationpageDe}%
	{\@declarationpageEn}
}


%% Assembly of title pages ----------------------------------------------------

\renewcommand{\@mastertitlepages}{
	\@mastertitle
	\@copyrightpage
	\@declarationpage
}

\renewcommand{\@diplomatitlepages}{
	\@diplomatitle
	\@copyrightpage
	\@declarationpage
}

\renewcommand{\@bachelortitlepages}{
	\@bachelortitle
	\@advisorpage
	\@declarationpage
}

\renewcommand{\@internshiptitlepages}{
	\@internshiptitle
	\@companypage
	\@declarationpage
}


% Add PDF meta data and create title pages

\renewcommand{\maketitle}{	%replace standard LaTeX-command
	\hypersetup{
			pdftitle={\@title},
			pdfauthor={\@author},
			pdfsubject={\@programname, \@placeofstudy},
			pdfcreator={LaTeX using class hgbthesis [\hgbDate]},
			pdfproducer={pdflatex},
			pdfkeywords={}
	}%

	\@maketitlepages
}

%% Additional Hagenberg packages ----------------------------------------------

\RequirePackage{latex/hgbmath}
\RequirePackage{latex/hgbheadings}
\RequirePackage{latex/hgbabbrev}
\RequirePackage{latex/hgblistings}
\RequirePackage{latex/hgbbib}
\RequirePackage{latex/user}


%% packages


\endinput
